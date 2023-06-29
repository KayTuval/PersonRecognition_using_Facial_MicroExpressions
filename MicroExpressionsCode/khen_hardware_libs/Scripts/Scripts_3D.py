__author__ = 'Khen Cohen'
__credits__ = ['Khen Cohen']
__email__ = 'khencohen@mail.tau.ac.il'
__date__ = '5.10.2021'

# from Scripts.Scripts import *
import datetime
from scipy import signal
import cv2 as cv
import numpy as np
import os
import sys
from scipy.interpolate import griddata
import matplotlib.pyplot as plt

####################################
######   Point Cloud Scripts  ######
####################################


def display_3d_scatter(x_points, y_points, z_points, marker = 'o'):
    plt.clf()
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x_points, y_points, z_points, marker=marker)
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')

    plt.show()
    return



# from pyntcloud import PyntCloud
# import pyvista as pv
# Show point cloud from 4D image format
def show_point_cloud_file(ply_filename, alpha = 1, show_edges = False):
    mesh = pv.read(ply_filename+".ply")
    mesh.delaunay_3d(alpha = alpha).plot(show_edges=show_edges, rgb=True)    # alpha = 0, tol  = 0.001
    # pc = PyntCloud.from_file(ply_filename+".ply")
    # pc.plot()


# # Create point cloud file in ply format
def generate_point_cloud_file_img_4d(img_4d, ply_filename, norm_factor, show = 0):
    height, width, __ = np.shape(img_4d)
    points = []
    for x in range(width):
        for y in range(height):
            if img_4d[y,x,3] == 0:
                continue
            disparity = np.divide(norm_factor, img_4d[y,x,3])
            points.append("%f %f %f %d %d %d\n" % (float(x), float(y), disparity, img_4d[y,x,0], img_4d[y,x,1], img_4d[y,x,2]))
            # points.append("%f %f %f %d %d %d 0\n" % (float(x), float(y), img_4d[y,x,3]/focal_length, img_4d[y,x,0], img_4d[y,x,2],img_4d[y,x,1]))
            #property uchar alpha
            # print(points[-1])

    file = open(ply_filename+'.ply', "w")
    file.write('''ply
format ascii 1.0
element vertex %d
property float x
property float y
property float z
property uchar red
property uchar green
property uchar blue
end_header
%s
''' % (len(points), "".join(points)))
    file.close()
    if show == 1:
        show_point_cloud_file(ply_filename)
    return



def write_ply(fn, verts, colors):
    ply_header = '''ply
format ascii 1.0
element vertex %(vert_num)d
property float x
property float y
property float z
property uchar red
property uchar green
property uchar blue
end_header
'''
    verts = verts.reshape(-1, 3)
    colors = colors.reshape(-1, 3)
    verts = np.hstack([verts, colors])
    with open(fn, 'w') as f:
        f.write(ply_header % dict(vert_num=len(verts)))
        np.savetxt(f, verts, '%f %f %f %d %d %d')

# Create point cloud file in ply format
def generate_point_cloud_file(points, colors, ply_filename, show = 0):
    write_ply(ply_filename+'.ply', points, colors)
    print('%s saved' % ply_filename+'.ply')
    if show == 1:
        show_point_cloud_file(ply_filename)

    return
