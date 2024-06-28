import numpy
def rot_points(xy,angle):
    rotMat = numpy.array([[numpy.cos(angle),-numpy.sin(angle)],
                          [numpy.sin(angle), numpy.cos(angle)]])
    points = numpy.matmul(rotMat ,xy)
    return points