// Copyright 2020 Intel Corporation

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassOptions.h"
#include "mlir/Support/DebugStringHelper.h"

// TODO: Including autotile.h for PowerOfTwoGenerator, but maybe instead both
// should include a third file with the tile size generators
#include "pmlc/dialect/pxa/analysis/strides.h"
#include "pmlc/dialect/pxa/ir/ops.h"
#include "pmlc/dialect/pxa/transforms/autotile.h"
#include "pmlc/dialect/pxa/transforms/passes.h"
#include "pmlc/dialect/pxa/transforms/stencil.h"

#include "pmlc/util/logging.h"

using namespace mlir; // NOLINT

namespace pmlc::dialect::pxa {

static AffineMap makeTileMap(MLIRContext *context, AffineMap map,
                             ValueRange operands,
                             ArrayRef<BlockArgument> idxs) {
  SmallVector<AffineExpr, 8> exprs;
  for (auto value : operands) {
    bool found = false;
    for (size_t i = 0; i < idxs.size(); i++) {
      if (value == idxs[i]) {
        exprs.push_back(getAffineDimExpr(i, context));
        found = true;
      }
    }
    if (!found) {
      exprs.push_back(getAffineConstantExpr(0, context));
    }
  }
  auto toIdxs = AffineMap::get(idxs.size(), 0, exprs, context);
  return map.compose(toIdxs);
}

struct BRGemmOperand {
  Value memref;
  AffineMap accessMap;
  AffineMap tileMap;

  template <typename TOp>
  BRGemmOperand(TOp op, ArrayRef<BlockArgument> idxs,
                SmallVectorImpl<Value> &mapOperands)
      : memref(op.getMemRef()), accessMap(op.getAffineMap()),
        tileMap(makeTileMap(op.getContext(), op.getAffineMap(),
                            op.getMapOperands(), idxs)) {
    mapOperands.append(op.getMapOperands().begin(), op.getMapOperands().end());
  }
};

class StencilBRGEMM : public StencilBase {
private:
  unsigned numThreads;
  StencilCostFunction stencilCostFn;

  Optional<LoadStoreOps> capture() {
    // Looking for load..load..mul..reduce..terminator
    LoadStoreOps ret;
    const unsigned kNumValidInstrInGemmRegion = 5;
    auto *body = op.getBody();

    // Verify the number of ops
    if (body->getOperations().size() != kNumValidInstrInGemmRegion) {
      IVLOG(5, "The AffineParallelOp region didn't have the right number of "
               "instructions for a GEMM");
      return llvm::None;
    }

    // Find the Reduce Op
    auto it = std::prev(body->end(), 2);
    auto reduceOp = dyn_cast<PxaReduceOp>(*it);
    if (!reduceOp) {
      IVLOG(5, "The AffineParallelOp region didn't have a reduce as its last "
               "non-terminator");
      return llvm::None;
    }
    ret.stores.push_back(&*it);
    IVLOG(5, "Found ReduceOp");

    // Now check the reduceOp aggregation.
    if (reduceOp.agg() != AtomicRMWKind::addf) {
      IVLOG(5, "the reduce operation is not addition");
      return llvm::None;
    }

    // Get the operand for the reduce op and make sure it is the result of a
    // multiplication.
    auto defOp = reduceOp.val().getDefiningOp();
    if (!defOp) {
      IVLOG(5,
            "the source of the reduce operation is not defined in this block");
      return llvm::None;
    }

    Operation *lhs;
    Operation *rhs;
    if (auto mulfOp = dyn_cast_or_null<MulFOp>(defOp)) {
      lhs = mulfOp.lhs().getDefiningOp();
      if (!dyn_cast_or_null<PxaLoadOp>(lhs)) {
        IVLOG(3, "The LHS of the mul op is not affine.load.");
        return llvm::None;
      }
      rhs = mulfOp.rhs().getDefiningOp();
      if (!dyn_cast_or_null<PxaLoadOp>(rhs)) {
        IVLOG(3, "The RHS of the mul op is not affine.load.");
        return llvm::None;
      }
    } else if (auto muliOp = dyn_cast_or_null<MulIOp>(defOp)) {
      lhs = muliOp.lhs().getDefiningOp();
      if (!dyn_cast_or_null<PxaLoadOp>(lhs)) {
        IVLOG(3, "The LHS of the mul op is not affine.load.");
        return llvm::None;
      }
      rhs = muliOp.rhs().getDefiningOp();
      if (!dyn_cast_or_null<PxaLoadOp>(rhs)) {
        IVLOG(3, "The RHS of the mul op is not affine.load.");
        return llvm::None;
      }
    } else {
      IVLOG(5, "The source of the reduce is not a multiplication operation");
      return llvm::None;
    }
    ret.loads.push_back(lhs);
    ret.loads.push_back(rhs);

    return Optional<LoadStoreOps>(ret);
  }

  double getCost(TensorAndIndexPermutation perm, ArrayRef<int64_t> tileSize) {
    unsigned tot_inner_loop = tileSize[0] * tileSize[1] * tileSize[2];

    auto cost = stencilCostFn(tileSize);
    if (cost.throughput == 0) {
      return std::numeric_limits<double>::infinity();
    }
    double inner_time = tot_inner_loop / cost.throughput;
    IVLOG(6,
          "Inner: loop = " << tot_inner_loop << " inner_time = " << inner_time);
    for (unsigned i = 0; i < getTiledIdxCount(); ++i) {
      IVLOG(6, debugString(perm.indexes[i]) << ": " << tileSize[i]);
    }

    // The middle idxs are the accumulation indexes, i.e. those used on loads
    // but not stores
    DenseMap<BlockArgument, unsigned> middle_idxs;
    auto in0StrideInfo = getStrideInfo(perm.ioOps[1]);
    for (const auto &kvp : in0StrideInfo->strides) {
      if (getBlockArgsAsSet().count(kvp.first)) {
        IVLOG(6, "Based on first tensor, inserting middle index "
                     << kvp.first.getArgNumber());
        middle_idxs.insert(std::make_pair(kvp.first, getIdxRange(kvp.first)));
      } else {
        IVLOG(5, "Index found from outside current loop on left input: "
                     << kvp.first.getArgNumber());
      }
    }
    IVLOG(5, "Current size of middle_idxs = " << middle_idxs.size());

    auto in1StrideInfo = getStrideInfo(perm.ioOps[2]);
    for (const auto &kvp : in1StrideInfo->strides) {
      if (getBlockArgsAsSet().count(kvp.first)) {
        IVLOG(6, "Based on second tensor, inserting middle index "
                     << kvp.first.getArgNumber());
        middle_idxs.insert(std::make_pair(kvp.first, getIdxRange(kvp.first)));
      } else {
        IVLOG(5, "Index found from outside current loop on right input: "
                     << kvp.first.getArgNumber());
      }
    }
    IVLOG(5, "Current size of middle_idxs = " << middle_idxs.size());
    auto outStrideInfo = getStrideInfo(perm.ioOps[0]);
    for (const auto &kvp : outStrideInfo->strides) {
      if (getBlockArgsAsSet().count(kvp.first)) {
        auto it = middle_idxs.find(kvp.first);
        if (it != middle_idxs.end()) {
          IVLOG(6, "Based on output tensor, erasing middle index "
                       << it->first.getArgNumber());
          middle_idxs.erase(it);
        }
      } else {
        IVLOG(5, "Index found from outside current loop on output: "
                     << kvp.first.getArgNumber());
      }
    }

    for (unsigned i = 0; i < getTiledIdxCount(); ++i) {
      assert(getBlockArgsAsSet().count(perm.indexes[i]) &&
             "All tiled indexes must be introduced in current loop");
      auto it = middle_idxs.find(perm.indexes[i]);
      if (it != middle_idxs.end()) {
        it->second = llvm::divideCeil(it->second, tileSize[i]);
      }
    }
    unsigned tot_middle_loop = 1;
    for (auto &kvp : middle_idxs) {
      tot_middle_loop *= kvp.second;
    }

    IVLOG(4, "Middle: loop = " << tot_middle_loop);

    if (VLOG_IS_ON(4)) {
      for (auto &kvp : middle_idxs) {
        if (kvp.second > 1) {
          IVLOG(4, kvp.first.getArgNumber() << ": " << kvp.second);
        }
      }
    }

    DenseMap<BlockArgument, unsigned> outer_idxs;
    for (const auto &kvp : outStrideInfo->strides) {
      if (getBlockArgsAsSet().count(kvp.first)) {
        IVLOG(4, "First: " << kvp.first.getArgNumber());
        IVLOG(5, "Second: " << kvp.second);
        IVLOG(5, "IdxRange: " << getIdxRange(kvp.first));
        outer_idxs.try_emplace(kvp.first, getIdxRange(kvp.first));
        IVLOG(4, "And now emplaced");
      }
      // If the index is from outside `op` this has already been logged, no need
      // for `else` branch here
    }
    for (unsigned i = 0; i < getTiledIdxCount(); i++) {
      assert(getBlockArgsAsSet().count(perm.indexes[i]) &&
             "All tiled indexes must be introduced in current loop");
      auto it = outer_idxs.find(perm.indexes[i]);
      if (it != outer_idxs.end()) {
        it->second = llvm::divideCeil(it->second, tileSize[i]);
      }
    }
    unsigned tot_outer_loop = 1;
    for (auto &kvp : outer_idxs) {
      tot_outer_loop *= kvp.second;
    }

    IVLOG(4, "Outer: loop = " << tot_outer_loop);

    if (VLOG_IS_ON(4)) {
      for (auto &kvp : outer_idxs) {
        if (kvp.second > 1) {
          IVLOG(4, kvp.first.getArgNumber() << ": " << kvp.second);
        }
      }
    }

    unsigned outer_batches = (tot_outer_loop - 1) / numThreads + 1;
    double perf =
        outer_batches * tot_middle_loop * (cost.startupCost + inner_time);

    IVLOG(3, "Performance = " << perf << "(outer count: " << outer_batches
                              << ", middle count: " << tot_middle_loop
                              << ", startup cost: " << cost.startupCost
                              << ", inner time: " << inner_time << ")");
    return perf;
  }

  void transform(TensorAndIndexPermutation perm, ArrayRef<int64_t> tileSize) {
    auto AStrideInfo = getStrideInfo(perm.ioOps[1]);
    int count = 0;
    int64_t kRange = -1, l_br = -1;
    for (const auto &kvp : AStrideInfo->strides) {
      count++;
      if (count == 2) {
        IVLOG(3, "AStride range: " << getIdxRange(kvp.first))
        kRange = getIdxRange(kvp.first);
      }
    }

    // First, modify step size of all tiled indexes
    auto steps = op.getSteps();
    for (size_t i = 0; i < getBlockArgsAsSet().size(); i++) {
      for (size_t j = 0; j < getTiledIdxCount(); j++) {
        if (perm.indexes[j] == op.getBody()->getArgument(i)) {
          steps[i] *= tileSize[j];

          // K index
          if (j == 2) {
            IVLOG(3, "steps[" << i << "] = " << steps[i]);
            if (kRange != -1) {
              l_br = kRange / steps[i];
              steps[i] = kRange;
            }
          }
        }
      }
    }
    op.setSteps(steps);

    // Generate the GEMM op; select inputs based on permutation order
    auto opC = cast<PxaReduceOp>(*perm.ioOps[0]);
    auto opA = cast<PxaLoadOp>(*perm.ioOps[1]);
    auto opB = cast<PxaLoadOp>(*perm.ioOps[2]);

    auto bodyBuilder = op.getBodyBuilder();

    auto tileAttr = bodyBuilder.getI64ArrayAttr(tileSize);
    auto lBrAttr = bodyBuilder.getI64IntegerAttr(l_br);

    SmallVector<Value, 8> mapOperands;
    BRGemmOperand c(opC, {perm.indexes[0], perm.indexes[1]}, mapOperands);
    BRGemmOperand a(opA, {perm.indexes[0], perm.indexes[2]}, mapOperands);
    BRGemmOperand b(opB, {perm.indexes[2], perm.indexes[1]}, mapOperands);

    auto brgemm = bodyBuilder.create<pxa::PxaBRGemmOp>(
        op.getLoc(), c.memref.getType(), //
        c.memref, AffineMapAttr::get(c.accessMap),
        AffineMapAttr::get(c.tileMap), //
        a.memref, AffineMapAttr::get(a.accessMap),
        AffineMapAttr::get(a.tileMap), //
        b.memref, AffineMapAttr::get(b.accessMap),
        AffineMapAttr::get(b.tileMap), //
        tileAttr, lBrAttr, mapOperands);

    opC.result().replaceAllUsesWith(brgemm);
    opC.erase();
  }

public:
  StencilBRGEMM(AffineParallelOp op, unsigned numThreads,
                StencilCostFunction costFn)
      : StencilBase{op,
                    3, // Three tileable indexes
                    {EvenTilingGenerator(), EvenTilingGenerator(),
                     EvenTilingGenerator()},
                    {IdxStrideReqs{
                         [](int64_t stride) { return stride != 0; }, // output
                         [](int64_t stride) { return stride != 0; }, // input0
                         [](int64_t stride) { return stride == 0; }, // input1
                     },
                     IdxStrideReqs{
                         [](int64_t stride) { return stride == 1; }, // output
                         [](int64_t stride) { return stride == 0; }, // input0
                         [](int64_t stride) { return stride == 1; }, // input1
                     },
                     IdxStrideReqs{
                         [](int64_t stride) { return stride == 0; }, // output
                         [](int64_t stride) { return stride == 1; }, // input0
                         [](int64_t stride) { return stride != 0; }, // input1
                     }}},
        numThreads{numThreads}, stencilCostFn(costFn) {}
};

struct StencilBRGEMMPass : public PassWrapper<StencilBRGEMMPass, FunctionPass> {
  StencilBRGEMMPass() {
    assert(false && "StencilBRGEMMPass must be configured");
  }

  StencilBRGEMMPass(const StencilBRGEMMPass &rhs) : costFn(rhs.costFn) {
    numThreads = rhs.numThreads.getValue();
  }

  StencilBRGEMMPass(unsigned numThreads_, StencilCostFunction costFn)
      : costFn(costFn) {
    numThreads = numThreads_;
  }

  void runOnFunction() final {
    auto func = getFunction();
    func.walk([this](AffineParallelOp op) {
      StencilBRGEMM stencil(op, numThreads.getValue(), costFn);
      stencil.DoStenciling();
    });
  }

  StencilCostFunction costFn;

  Option<unsigned> numThreads{
      *this, "threads",
      llvm::cl::desc("Specifies number of threads for the stencil pass")};
};

std::unique_ptr<Pass> createStencilBRGEMMPass(unsigned numThreads,
                                              StencilCostFunction costFn) {
  return std::make_unique<StencilBRGEMMPass>(numThreads, costFn);
}

} // namespace pmlc::dialect::pxa