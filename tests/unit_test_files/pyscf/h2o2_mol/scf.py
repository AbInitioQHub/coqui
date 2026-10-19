from pyscf import gto, df, scf, dft
import coqui.interaction.pyscf_interface.gdf_converter as gdf_converter
import coqui.mean_field.pyscf_interface.converter as mf_converter
import h5

mol = gto.M(atom='H -1 -1 0; O 0 0 0; O 1 0 0; H 1 0 1', basis='6-31g', verbose=7)

mydf = df.DF(mol, auxbasis='6-311++g')
mydf._cderi_to_save = 'cderi.h5'
mydf.build()
gdf_converter.mol_gdf_grad_dump_to_h5(mydf, mo=False, outdir='gdf_eri')

mf = scf.RHF(mol).density_fit()
mf.with_df = mydf
mf.kernel()
mf.analyze()
mf_converter.mol_dump_to_h5(mf, becke_grid_level=5, mo=False, outdir='./', prefix='pyscf')

mf_grad = mf.Gradients()
mf_grad.auxbasis_response = True
grad_total = mf_grad.kernel()

filename = './pyscf.h5'
with h5.HDFArchive(filename, 'a') as root:
    g = root['SCF']
    g['grad_total'] = grad_total
