load("@rules_cc//cc:defs.bzl", "cc_import")

def _sanitize_label(path):
    return path.replace("/", "_").replace(".", "_").replace("-", "_")

def hip_static_library(
        name,
        srcs,
        hdrs = [],
        visibility = None):
    object_labels = []

    for src in srcs:
        object_name = name + "_" + _sanitize_label(src) + "_object"
        object_file = object_name + ".o"

        native.genrule(
            name = object_name,
            srcs = [src] + hdrs,
            outs = [object_file],
            cmd = "hipcc -std=c++20 -fPIC -I. -Iinclude -I/opt/rocm/include -c $(location %s) -o $@" % src,
        )

        object_labels.append(":" + object_name)

    archive_name = name + "_archive"

    native.genrule(
        name = archive_name,
        srcs = object_labels,
        outs = ["lib%s.a" % name],
        cmd = "ar rcs $@ $(SRCS)",
    )

    cc_import(
        name = name,
        hdrs = hdrs,
        static_library = ":" + archive_name,
        visibility = visibility,
    )