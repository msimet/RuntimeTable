import numpy
import numpy.testing
try:
    from RuntimeTable import RuntimeTable
except ImportError:
    import sys
    sys.path.append('..')
    from RuntimeTable import RuntimeTable

x_boundaries = (0,7)
sparse_num = 8
fine_num = 7001
sparse_offset = 1.0*(x_boundaries[1]-x_boundaries[0])/(2*(sparse_num-1))
fine_offset = 1.0*(x_boundaries[1]-x_boundaries[0])/(2*(fine_num-1))
x_sparse_data = numpy.linspace(x_boundaries[0]+sparse_offset, x_boundaries[1]-sparse_offset, num=sparse_num-1)
x_fine_data = numpy.linspace(x_boundaries[0]+fine_offset, x_boundaries[1]-fine_offset, num=fine_num-1)

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
    
    sparse_floor = numpy.floor(x_sparse_data)
    fine_dx = x_fine_data[1]-x_fine_data[0]
    fine_floor = fine_dx*numpy.floor(x_fine_data/fine_dx)
    sparse_err = numpy.max(((sparse_floor+1)**2-sparse_floor**2)*(x_sparse_data-sparse_floor)+sparse_floor**2-x_sparse_data**2)
    fine_err = numpy.max(((fine_floor+fine_dx)**2-fine_floor**2)*(x_fine_data-fine_floor)/fine_dx+fine_floor**2-x_fine_data**2)
    sparse_decimal = int(numpy.floor(-numpy.log10(sparse_err)))
    fine_decimal = int(numpy.floor(-numpy.log10(fine_err)))
    
    numpy.testing.assert_almost_equal(sparse_table(x_sparse_data), x2(x_sparse_data), decimal=sparse_decimal)
    numpy.testing.assert_almost_equal(fine_table(x_fine_data), x2(x_fine_data), decimal=fine_decimal)
    
    numpy.testing.assert_raises(ValueError, sparse_table, x_boundaries[1]+1)
    numpy.testing.assert_raises(ValueError, fine_table, x_boundaries[0]-1) 
    
def test_2d():
    """Test that RuntimeTable produces proper results for multidimensional functions and fails properly for nonsensical inputs"""
    
    sparse_table_xy2 = RuntimeTable(xy2, boundaries=[x_boundaries]*2, nsteps=sparse_num)
    fine_table_xy2 = RuntimeTable(xy2, boundaries=[x_boundaries]*2, nsteps=[fine_num, fine_num])
    sparse_table_xyz2 = RuntimeTable(xyz2, boundaries=[x_boundaries]*3, nsteps=[sparse_num, sparse_num, sparse_num])
    fine_table_xyz2 = RuntimeTable(xyz2, boundaries=[x_boundaries]*3, nsteps=fine_num)
    
    sparse_2d = numpy.meshgrid(x_sparse_data, x_sparse_data)
    fine_2d = numpy.meshgrid(x_fine_data, x_fine_data)
    sparse_3d = numpy.meshgrid(x_sparse_data, x_sparse_data, x_sparse_data)
    fine_3d = numpy.meshgrid(x_fine_data, x_fine_data, x_fine_data)

    sparse_floor = numpy.floor(sparse_2d[0])
    fine_dx = x_fine_data[1]-x_fine_data[0]
    fine_floor = fine_dx*numpy.floor(fine_2d[0]/fine_dx)
    sparse_err = numpy.max(sparse_2d[1]*(((sparse_floor+1)**2-sparse_floor**2)*(sparse_2d[0]-sparse_floor)+sparse_floor**2-sparse_2d[0]**2))
    fine_err = numpy.max(fine_2d[1]*(((fine_floor+fine_dx)**2-fine_floor**2)*(fine_2d[0]-fine_floor)/fine_dx+fine_floor**2-fine_2d[0]**2))
    sparse_decimal = int(numpy.floor(-numpy.log10(sparse_err)))
    fine_decimal = int(numpy.floor(-numpy.log10(fine_err)))

    
    numpy.testing.assert_almost_equal(sparse_table_xy2([x_sparse_data]*2), xy2([x_sparse_data]*2))
    numpy.testing.assert_almost_equal(fine_table_xy2([x_fine_table]*2), xy2([x_fine_table]*2))
    
    sparse_floor = numpy.floor(sparse_3d[0])
    fine_dx = x_fine_data[1]-x_fine_data[0]
    fine_floor = fine_dx*numpy.floor(fine_3d[0]/fine_dx)
    sparse_err = numpy.max(sparse_3d[1]*sparse_3d[2]*(((sparse_floor+1)**2-sparse_floor**2)*(sparse_3d[0]-sparse_floor)+sparse_floor**2-sparse_3d[0]**2))
    fine_err = numpy.max(fine_2d[1]*fine_3d[2]*(((fine_floor+fine_dx)**2-fine_floor**2)*(fine_3d[0]-fine_floor)/fine_dx+fine_floor**2-fine_3d[0]**2))
    sparse_decimal = int(numpy.floor(-numpy.log10(sparse_err)))
    fine_decimal = int(numpy.floor(-numpy.log10(fine_err)))

    
    numpy.testing.assert_almost_equal(sparse_table_xyz2([x_sparse_data]*3), xyz2([x_sparse_data]*3))
    numpy.testing.assert_almost_equal(fine_table_xyz2([x_fine_table]*3), xyz2([x_fine_table]*3))    
    numpy.testing.assert_raises(RuntimeError, sparse_table_xy2([x_sparse_data]*3))
    numpy.testing.assert_raises(RuntimeError, fine_table_xyz2([x_sparse_data]*2))
    
    midpoint  = 0.5*(x_boundaries[0]+x_boundaries[1])
    numpy.testing.assert_raises(RuntimeError, sparse_table_xy2([midpoint, x_boundaries[1]+1]))
    numpy.testing.assert_raises(RuntimeError, fine_table_xy2([midpoint, x_boundaries[0]-1]))
    numpy.testing.assert_raises(RuntimeError, fine_table_xy2([x_boundaries[1]+1, midpoint]))
    numpy.testing.assert_raises(RuntimeError, sparse_table_xy2([x_boundaries[0]-1, midpoint]))
    numpy.testing.assert_raises(RuntimeError, sparse_table_xyz2([midpoint, midpoint, x_boundaries[1]+1]))
    numpy.testing.assert_raises(RuntimeError, fine_table_xyz2([midpoint, x_boundaries[0]-1, midpoint]))
    numpy.testing.assert_raises(RuntimeError, fine_table_xyz2([x_boundaries[1]+1, midpoint, midpoint]))
    numpy.testing.assert_raises(RuntimeError, sparse_table_xyz2([midpoint, x_boundaries[0]-1, midpoint]))
    numpy.testing.assert_raises(RuntimeError, sparse_table_xy2([x_boundaries[0]-1, x_boundaries[1]+1]))
    numpy.testing.assert_raises(RuntimeError, fine_table_xy2([x_boundaries[0]-1, x_boundaries[0]-1]))
    numpy.testing.assert_raises(RuntimeError, fine_table_xyz2([x_boundaries[1]+1, x_boundaries[1]+1, x_boundaries[0]-1]))
    numpy.testing.assert_raises(RuntimeError, sparse_table_xyz2([x_boundaries[0]-1, x_boundaries[0]-1, x_boundaries[0]-1]))

if __name__=='__main__':
    test_table_creation()
    test_1d()
    test_2d()
