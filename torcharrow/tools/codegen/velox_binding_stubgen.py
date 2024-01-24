# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import logging
import os

import torch  # noqa

from libfb.py.log import set_simple_logging
from pybind11_stubgen.stubgen import ChdirGuard, ModuleStubsGenerator


set_simple_logging()
logger: logging.Logger = logging.getLogger(__name__)


VELOX_LIB_NAME = "_torcharrow"
VELOX_LIB_DIR = "csrc/velox"
VELOX_LIB_MODULE_NAME = f"torcharrow.{VELOX_LIB_NAME}"
VELOX_LIB_STUB_NAME = f"{VELOX_LIB_NAME}.pyi"


def main() -> None:
    script_path = os.path.dirname(os.path.realpath(__file__))
    repo_path = os.path.abspath(os.path.join(script_path, "../../"))
    logger.info("Found repo checkout at %s", repo_path)
    output_dir = os.path.join(repo_path, VELOX_LIB_DIR)

    logger.info(
        "Will generate type stubs for module '%s' in '%s'",
        VELOX_LIB_MODULE_NAME,
        output_dir,
    )
    with ChdirGuard(output_dir):
        module = ModuleStubsGenerator(VELOX_LIB_MODULE_NAME)
        module.parse()
        module.write()

    # some hacks
    with open(os.path.join(output_dir, VELOX_LIB_STUB_NAME), "r+") as f:
        content = f.read()

        # You can't import yourself in a stubfile of the same name
        content = content.replace(f"import {VELOX_LIB_MODULE_NAME}\n", "")

        # This is a hacky way to put a warning in the file that it was generated
        f.seek(0)
        f.write(
            "# Copyright (c) Meta Platforms, Inc. and affiliates.\n"
            "# This file is generated by `tools/codegen/velox_binding_stubgen.py\n"
            "# and should not be edited manually.\n"
            "# \x40generated\n"
        )
        f.write(content)


if __name__ == "__main__":
    main()
