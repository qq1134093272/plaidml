load(":build_defs.bzl", "gcc_toolchain_config")

package(default_visibility = ["//visibility:public"])

config_setting(
    name = "linux_x86_64",
    values = {"cpu": "linux_x86_64"},
)

config_setting(
    name = "macos_x86_64",
    values = {"cpu": "darwin_x86_64"},
)

config_setting(
    name = "windows_x86_64",
    values = {"cpu": "x64_windows"},
)

cc_toolchain_suite(
    name = "toolchain",
    tags = ["linux"],
    # target_cpu | compiler
    toolchains = {
        "linux_x86_64|gcc": ":cc-linux_x86_64-gcc_8.3.0",
        "linux_x86_64": ":cc-linux_x86_64-gcc_8.3.0",
    },
)

filegroup(
    name = "empty",
    srcs = [],
)

cc_toolchain(
    name = "cc-linux_x86_64-gcc_8.3.0",
    all_files = "//toolchain/crosstool_ng/linux_x86_64-gcc_8.3.0:all_files",
    compiler_files = "//toolchain/crosstool_ng/linux_x86_64-gcc_8.3.0:compiler_files",
    dwp_files = ":empty",
    linker_files = "//toolchain/crosstool_ng/linux_x86_64-gcc_8.3.0:linker_files",
    objcopy_files = "//toolchain/crosstool_ng/linux_x86_64-gcc_8.3.0:objcopy",
    strip_files = "//toolchain/crosstool_ng/linux_x86_64-gcc_8.3.0:strip",
    supports_param_files = 1,
    tags = ["linux"],
    toolchain_config = ":cc_linux_x86_64-gcc_8.3.0",
    toolchain_identifier = "gcc-linux_x86_64",
)

gcc_toolchain_config(
    name = "cc_linux_x86_64-gcc_8.3.0",
    libc_version = "glibc_2.12.1",
    tags = ["linux"],
    target = "linux_x86_64",
    version = "8.3.0",
)
