from firedrake import *
import pytest
from petsc4py import PETSc
from petsc4py.PETSc import ViewerHDF5
from pyop2.mpi import COMM_WORLD


@pytest.mark.parallel(nprocs=4)
def test_io_hdf5_xdmf():
    # Parameters
    fname = "./test_io_mesh_dump.h5"
    outformat = ViewerHDF5.Format.HDF5_XDMF
    ntimes = 3
    heterogeneous = True
    # Create mesh.
    mesh = RectangleMesh(4, 1, 4., 1.)
    mesh.init()
    plex = mesh._topology_dm
    plex.setOptionsPrefix("original_")
    plex.viewFromOptions("-dm_view")
    # Save mesh.
    mesh.view(fname, outformat)
    # Load -> Save -> Load ...
    grank = COMM_WORLD.rank
    for i in range(ntimes):
        if heterogeneous:
            mycolor = (grank > ntimes - i)
        else:
            mycolor = 0
        comm = COMM_WORLD.Split(color=mycolor, key=grank)
        if mycolor == 0:
            PETSc.Sys.Print("Begin cycle %d" % i, comm=comm)
            # Load.
            mesh = Mesh(fname, comm=comm)
            mesh.init()
            # Test if DM is distributed.
            flg = mesh._topology_dm.isDistributed()
            PETSc.Sys.Print("Loaded mesh distributed? %s" % flg, comm=comm)
            # Save.
            mesh.view(fname, outformat)
            PETSc.Sys.Print("End   cycle %d\n--------\n" % i, comm=comm)
        COMM_WORLD.Barrier()
