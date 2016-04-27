import numpy
import numpy.testing
try:
    from RuntimeTable import RuntimeTable, stack
except ImportError:
    import sys
    sys.path.append('..')
    from RuntimeTable import RuntimeTable, stack

# Test data set parameters
x_boundaries = (0,7)
sparse_num = 8
fine_num = 71
# Intermediate points at which to test interpolation
sparse_offset = 1.0*(x_boundaries[1]-x_boundaries[0])/(2*(sparse_num-1))
fine_offset = 1.0*(x_boundaries[1]-x_boundaries[0])/(2*(fine_num-1))
x_sparse_data = numpy.linspace(x_boundaries[0]+sparse_offset, x_boundaries[1]-sparse_offset, num=sparse_num-1)
x_fine_data = numpy.linspace(x_boundaries[0]+fine_offset, x_boundaries[1]-fine_offset, num=fine_num-1)
fine_dx = x_fine_data[1]-x_fine_data[0]
sparse_dx = x_sparse_data[1]-x_sparse_data[0]

def x2(x):
    return x*x

def xy2(x, y):
    return x*y*y

def xyz2(x, y, z):
    return x*y*z*z

def test_table_creation():
    """Test that tables can be created, or errors properly raised when incorrect arguments are given."""
    numpy.testing.assert_raises(TypeError, RuntimeTable)
    numpy.testing.assert_raises(RuntimeError, RuntimeTable, [1,2,3,4,5])
    numpy.testing.assert_raises(RuntimeError, RuntimeTable, [1,2,3,4,5], 10)
    numpy.testing.assert_raises(RuntimeError, RuntimeTable, None, boundaries=x_boundaries, nsteps=sparse_num)
    numpy.testing.assert_raises(RuntimeError, RuntimeTable, x2, boundaries=x_boundaries)
    numpy.testing.assert_raises(RuntimeError, RuntimeTable, x2, boundaries=[x_boundaries, x_boundaries])
    numpy.testing.assert_raises(RuntimeError, RuntimeTable, x2, nsteps=sparse_num)
    numpy.testing.assert_raises(RuntimeError, RuntimeTable, x2, nsteps=[sparse_num, fine_num])
    numpy.testing.assert_raises(RuntimeError, RuntimeTable, x2, boundaries=x_boundaries, nsteps=[sparse_num, fine_num])
    # This should pass, using the same nsteps for all!
    RuntimeTable(xy2, boundaries=[x_boundaries, x_boundaries], nsteps=sparse_num)
    RuntimeTable(xy2, boundaries=[x_boundaries, x_boundaries], nsteps=[sparse_num, fine_num])
    RuntimeTable(x2, boundaries=x_boundaries, nsteps=sparse_num)

    
def test_1d():
    """Test that RuntimeTable produces proper results for single-variable functions and fails properly for
    nonsensical inputs."""
    sparse_table = RuntimeTable(x2, boundaries=x_boundaries, nsteps=sparse_num)
    fine_table = RuntimeTable(x2, boundaries=x_boundaries, nsteps=fine_num)
    
    # The error from a linear interpolation of a polynomial is pretty straightforward; we compute it
    # here to set the precision of our almost-equal testing later.
    sparse_floor = sparse_dx*numpy.floor(x_sparse_data/sparse_dx)
    fine_floor = fine_dx*numpy.floor(x_fine_data/fine_dx)
    sparse_err = numpy.max(((sparse_floor+1)**2-sparse_floor**2)*(x_sparse_data-sparse_floor)/sparse_dx+sparse_floor**2-x_sparse_data**2)
    fine_err = numpy.max(((fine_floor+fine_dx)**2-fine_floor**2)*(x_fine_data-fine_floor)/fine_dx+fine_floor**2-x_fine_data**2)
    sparse_decimal = int(numpy.floor(-numpy.log10(sparse_err)))
    fine_decimal = int(numpy.floor(-numpy.log10(fine_err)))
    
    numpy.testing.assert_almost_equal(sparse_table(x_sparse_data), x2(x_sparse_data), decimal=sparse_decimal)
    numpy.testing.assert_almost_equal(fine_table(x_fine_data), x2(x_fine_data), decimal=fine_decimal)
    
    # Out of bounds errors
    numpy.testing.assert_raises(ValueError, sparse_table, x_boundaries[1]+1)
    numpy.testing.assert_raises(ValueError, fine_table, x_boundaries[0]-1) 
    
def test_2d():
    """Test that RuntimeTable produces proper results for multidimensional functions and fails properly for nonsensical inputs"""
    
    sparse_table_xy2 = RuntimeTable(xy2, boundaries=[x_boundaries]*2, nsteps=sparse_num)
    fine_table_xy2 = RuntimeTable(xy2, boundaries=[x_boundaries]*2, nsteps=[fine_num, fine_num])
    sparse_table_xyz2 = RuntimeTable(xyz2, boundaries=[x_boundaries]*3, nsteps=[sparse_num, sparse_num, sparse_num])
    fine_table_xyz2 = RuntimeTable(xyz2, boundaries=[x_boundaries]*3, nsteps=fine_num)
    
    sparse_2d = numpy.asarray([x_sparse_data]*2).transpose()
    fine_2d = numpy.asarray([x_fine_data]*2).transpose()
    sparse_3d = numpy.asarray([x_sparse_data]*3).transpose()
    fine_3d = numpy.asarray([x_fine_data]*3).transpose()

    # First, test 2d inputs with the coordinates given as an x vector and a y vector
    sparse_floor = sparse_dx*numpy.floor(sparse_2d[:,0]/sparse_dx)
    fine_floor = fine_dx*numpy.floor(fine_2d[:,0]/fine_dx)
    sparse_err = numpy.max(sparse_2d[:,1]*(((sparse_floor+sparse_dx)**2-sparse_floor**2)*(sparse_2d[:,0]-sparse_floor)/sparse_dx+sparse_floor**2-sparse_2d[:,0]**2))
    fine_err = numpy.max(fine_2d[:,1]*(((fine_floor+fine_dx)**2-fine_floor**2)*(fine_2d[:,0]-fine_floor)/fine_dx+fine_floor**2-fine_2d[:,0]**2))
    sparse_decimal = int(numpy.floor(-numpy.log10(sparse_err)))
    fine_decimal = int(numpy.floor(-numpy.log10(fine_err)))
    
    numpy.testing.assert_almost_equal(sparse_table_xy2([x_sparse_data]*2), xy2(*[x_sparse_data]*2), decimal=sparse_decimal)
    numpy.testing.assert_almost_equal(fine_table_xy2([x_fine_data]*2), xy2(*[x_fine_data]*2), decimal=fine_decimal)
    
    # Now, test 2d inputs with the coordinates given as a multidimensional array
    sparse_2d_grid = numpy.meshgrid(x_sparse_data, x_sparse_data)
    fine_2d_grid = numpy.meshgrid(x_fine_data, x_fine_data)
    sparse_3d_grid = numpy.meshgrid(x_sparse_data, x_sparse_data, x_sparse_data)
    fine_3d_grid = numpy.meshgrid(x_fine_data, x_fine_data, x_fine_data)
    sparse_floor = sparse_dx*numpy.floor(sparse_2d_grid[0]/sparse_dx)
    fine_floor = fine_dx*numpy.floor(fine_2d_grid[0]/fine_dx)
    sparse_err = numpy.max(sparse_2d_grid[1]*(((sparse_floor+sparse_dx)**2-sparse_floor**2)*(sparse_2d_grid[0]-sparse_floor)/sparse_dx+sparse_floor**2-sparse_2d_grid[0]**2))
    fine_err = numpy.max(fine_2d_grid[1]*(((fine_floor+fine_dx)**2-fine_floor**2)*(fine_2d_grid[0]-fine_floor)/fine_dx+fine_floor**2-fine_2d_grid[0]**2))
    sparse_decimal = int(numpy.floor(-numpy.log10(sparse_err)))
    fine_decimal = int(numpy.floor(-numpy.log10(fine_err)))

    numpy.testing.assert_almost_equal(sparse_table_xy2(sparse_2d_grid), xy2(*sparse_2d_grid), decimal=sparse_decimal)
    numpy.testing.assert_almost_equal(fine_table_xy2(fine_2d_grid), xy2(*fine_2d_grid), decimal=fine_decimal)

    combo_sparse_2d_grid = numpy.dstack(sparse_2d_grid)
    combo_fine_2d_grid = numpy.dstack(fine_2d_grid)
    numpy.testing.assert_almost_equal(sparse_table_xy2(combo_sparse_2d_grid), xy2(*sparse_2d_grid), decimal=sparse_decimal)
    numpy.testing.assert_almost_equal(fine_table_xy2(combo_fine_2d_grid), xy2(*fine_2d_grid), decimal=fine_decimal)
         
 
    # Three-dimensional tests
    sparse_floor = sparse_dx*numpy.floor(sparse_3d[:,0]/sparse_dx)
    fine_floor = fine_dx*numpy.floor(fine_3d[:,0]/fine_dx)
    sparse_err = numpy.max(sparse_3d[:,1]*sparse_3d[:,2]*(((sparse_floor+1)**2-sparse_floor**2)*(sparse_3d[:,0]-sparse_floor)/sparse_dx+sparse_floor**2-sparse_3d[:,0]**2))
    fine_err = numpy.max(fine_2d[:,1]*fine_3d[:,2]*(((fine_floor+fine_dx)**2-fine_floor**2)*(fine_3d[:,0]-fine_floor)/fine_dx+fine_floor**2-fine_3d[:,0]**2))
    sparse_decimal = int(numpy.floor(-numpy.log10(sparse_err)))
    fine_decimal = int(numpy.floor(-numpy.log10(fine_err)))

    
    numpy.testing.assert_almost_equal(sparse_table_xyz2([x_sparse_data]*3), xyz2(*[x_sparse_data]*3), decimal=sparse_decimal)
    numpy.testing.assert_almost_equal(fine_table_xyz2([x_fine_data]*3), xyz2(*[x_fine_data]*3), decimal=fine_decimal)    
    combo_sparse_3d_grid = stack(*sparse_3d_grid)
    combo_fine_3d_grid = stack(*fine_3d_grid)
    numpy.testing.assert_almost_equal(sparse_table_xyz2(combo_sparse_3d_grid), xyz2(*sparse_3d_grid), decimal=sparse_decimal)
    numpy.testing.assert_almost_equal(fine_table_xyz2(combo_fine_3d_grid), xyz2(*fine_3d_grid), decimal=fine_decimal)


    # Wrong number of input dimensions
    # TODO: grid inputs
    numpy.testing.assert_raises(TypeError, sparse_table_xy2, [x_sparse_data]*3)
    numpy.testing.assert_raises(TypeError, fine_table_xyz2, [x_sparse_data]*2)
    
    # Out of bounds errors
    midpoint  = 0.5*(x_boundaries[0]+x_boundaries[1])
    numpy.testing.assert_raises(ValueError, sparse_table_xy2, [midpoint, x_boundaries[1]+1])
    numpy.testing.assert_raises(ValueError, fine_table_xy2, [midpoint, x_boundaries[0]-1])
    numpy.testing.assert_raises(ValueError, fine_table_xy2, [x_boundaries[1]+1, midpoint])
    numpy.testing.assert_raises(ValueError, sparse_table_xy2, [x_boundaries[0]-1, midpoint])
    numpy.testing.assert_raises(ValueError, sparse_table_xyz2, [midpoint, midpoint, x_boundaries[1]+1])
    numpy.testing.assert_raises(ValueError, fine_table_xyz2, [midpoint, x_boundaries[0]-1, midpoint])
    numpy.testing.assert_raises(ValueError, fine_table_xyz2, [x_boundaries[1]+1, midpoint, midpoint])
    numpy.testing.assert_raises(ValueError, sparse_table_xyz2, [midpoint, x_boundaries[0]-1, midpoint])
    numpy.testing.assert_raises(ValueError, sparse_table_xy2, [x_boundaries[0]-1, x_boundaries[1]+1])
    numpy.testing.assert_raises(ValueError, fine_table_xy2, [x_boundaries[0]-1, x_boundaries[0]-1])
    numpy.testing.assert_raises(ValueError, fine_table_xyz2, [x_boundaries[1]+1, x_boundaries[1]+1, x_boundaries[0]-1])
    numpy.testing.assert_raises(ValueError, sparse_table_xyz2, [x_boundaries[0]-1, x_boundaries[0]-1, x_boundaries[0]-1])

if __name__=='__main__':
    test_table_creation()
    test_1d()
    test_2d()
