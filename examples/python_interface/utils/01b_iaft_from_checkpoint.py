"""
Example: Constructing IAFT from a CoQuí checkpoint file

This example shows how to construct an IAFT instance directly from a CoQui checkpoint h5. 
This is useful for ensuring the IAFT instance constructed has he exact same parameters 
(inverse temperature, frequency cutoff, precision) that were used in the CoQui calculation
where the checkpoint was generated.

See also
--------
- IAFT basics: 01a_iaft_basics.py
"""

import h5py
import coqui
from coqui.utils.imag_axes_ft import IAFT

checkpoint_h5 = coqui.TEST_INPUT_DIR + "qe/svo_kp222_nbnd40/gw/svo.mbpt.h5"

# Load IAFT from checkpoint
iaft = IAFT.from_coqui_chkpt(checkpoint_h5, verbose=True)