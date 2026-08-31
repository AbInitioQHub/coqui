"""
==========================================================================
CoQuí: Correlated Quantum ínterface

Copyright (c) 2022-2026 Simons Foundation & The CoQuí developer team

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==========================================================================
"""

import h5._h5py as h5
import argparse

# Usage: 
# python3 read_last_it.py --inp mbpt.gf2.iterative.mbpt.h5  --out mbpt.gf2.last.mbpt.h5 --keep_first_it True
# Attention: may not work correctly if h5 file is corrupted

parser = argparse.ArgumentParser(description="Script to read the last iteraction")
parser.add_argument("--inp", type=str, default="mbpt.h5", help="file to be read")
parser.add_argument("--out", type=str, default="mbpt_last.h5", help="output file")
parser.add_argument("--keep_first_it", action="store_true", help="keep the first iteration")

args = parser.parse_args()
finput = args.inp
foutput = args.out
keep_first_it = args.keep_first_it

h_in = h5.File(finput, 'r')
h_out = h5.File(foutput, 'w')


def recursive_copy(g_in, g_out):
    for k in g_in.keys():
        copy_from_key(g_in, g_out, k)


def copy_from_key(g_in, g_out, k):
    # recursively copy subgroups without datasets
    if g_in.has_subgroup(k):
        g_out.create_group(k)
        gg_in = g_in.open_group(k)
        gg_out = g_out.open_group(k)
        recursive_copy(gg_in, gg_out)
        # copy datasets (do no occur in the current format)
    else:
        if g_in.has_dataset(k):
            h5.h5_write(g_out, k, h5.h5_read(g_in, k))
        else:
            raise ValueError("The key is neither group nor dataset")


def copy_all_and_last_iters(g_in, g_out, keep_first_iter=False):
    for k in g_in.keys():
        if k not in {'scf', 'embed'}:
            copy_from_key(g_in, g_out, k)
        else:
            gg_in = g_in.open_group(k)
            g_out.create_group(k)
            gg_out = g_out.open_group(k)
            if gg_in.has_dataset('final_iter'):
                final_iter = h5.h5_read(gg_in, 'final_iter')
                h5.h5_write(gg_out, 'final_iter', final_iter)
                # copy the last iteration
                if gg_in.has_subgroup('iter'+str(final_iter)):
                    copy_from_key(gg_in, gg_out, 'iter'+str(final_iter))
                if keep_first_iter and final_iter!=1:
                    if gg_in.has_subgroup('iter1'):
                        copy_from_key(gg_in, gg_out, 'iter1')
            else:
                raise ValueError("The key final_iter is not found!")
                

g_in = h5.Group(h_in)
g_out = h5.Group(h_out)

copy_all_and_last_iters(g_in, g_out, keep_first_it)

del h_out
del h_in
