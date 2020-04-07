#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rospy
import numpy as np
from scipy.spatial import ConvexHull
from sklearn.cluster import DBSCAN
from jsk_recognition_msgs.msg import ModelCoefficientsArray
from jsk_recognition_msgs.msg import PolygonArray
from std_msgs.msg import ColorRGBA
from geometry_msgs.msg import Pose
from geometry_msgs.msg import Polygon
from sensor_msgs.msg import PointCloud2
from geometry_msgs.msg import Point32
from std_msgs.msg import Float32MultiArray
from geometry_msgs.msg import PolygonStamped
from visualization_msgs.msg import Marker
from std_msgs.msg import Header
import sensor_msgs.point_cloud2 as pc2
from sensor_msgs.msg import CameraInfo
from visualization_msgs.msg import MarkerArray
import message_filters, tf
from copy import deepcopy as dcopy


from sensor_msgs.msg import PointCloud2
from std_msgs.msg import Int8MultiArray

class DetectorDepth:
    def __init__(self):
        rospy.init_node("depth_stair_detector")
        print("Node initialization")
        self.stair_limits = {'depth': .3, 'height': .5}
        # rospy.Subscriber('/multi_plane_estimate/output_coefficients', ModelCoefficientsArray, self.callback)
        # rospy.Subscriber('/multi_plane_estimate/output_polygon', PolygonArray, self.callback)
        self.floor_pub = rospy.Publisher('/floor', PolygonArray)
        self.stair_pub = rospy.Publisher('/stair', PolygonArray)
        self.marker_array = MarkerArray()
        self.marker = Marker()
        self.br = tf.TransformBroadcaster()
        self.stair_plane_header = Header()
        self.stair_plane_header.frame_id = 'camera_link'
        self.polygon_array_msg = PolygonArray()
        coefs = message_filters.Subscriber('/multi_plane_estimate/output_refined_coefficients', ModelCoefficientsArray)
        polygs = message_filters.Subscriber('/multi_plane_estimate/output_refined_polygon', PolygonArray)
        point_cloud = message_filters.Subscriber('/plane_extraction/output_nonplane_cloud', PointCloud2)
        self.resolution = self.init_depth_camera(rospy.wait_for_message("camera/depth/camera_info", CameraInfo))
        # depth = message_filters.Subscriber('/multi_plane_estimate/output_refined_polygon', PolygonArray)
        ts = message_filters.TimeSynchronizer([coefs, polygs, point_cloud], 10)
        ts.registerCallback(self.callback)
        self.array_pub = rospy.Publisher('/stair/parameters', Float32MultiArray)
        self.corner_pub = rospy.Publisher('/stair/bbox/depth', Float32MultiArray)
        # Polygon array
        self.stair_plane_pub = rospy.Publisher('/stair/plane', PolygonArray)
        rospy.spin()

    def init_depth_camera(self, msg):
        return {
            'height': msg.height,
            'width': msg.width,
            'H': 60.0/180*3.14,
            'V': 49.5 / 180 * 3.14,
            'D': 73 / 180 * 3.14
        }

    def polygons2list(self, polygs):
        planes = []
        for plane in polygs.polygons:
            cur = []
            for point in plane.polygon.points:
                x = point.x
                y = point.y
                z = point.z
                cur.append([x, y, z])
            planes.append(cur)
        return planes

    def coefs2list(self, coefs):
        return [plane.values for plane in coefs.coefficients]

    def get_perimeters(self, planes):
        perimeters = []
        for plane in planes:
            perim = 0
            for i in range(1, len(plane)):
                p_ = plane[i-1]
                p = plane[i]
                # distance between two points
                perim += ((p[0] - p_[0] )**2 + (p[1] - p_[1] )**2 + (p[2] - p_[2] )**2)**0.5
            perimeters.append(perim)
        return perimeters

    def find_perpendicularity(self, coefs):
        length = len(coefs)
        P = []
        for i in range(length):
            cur = []
            for j in range(length):
                # coefs[i] - current plane
                # we are looking for perpendicularity with jth plane
                cur.append(round(np.dot(coefs[i][:3], coefs[j][:3]), 2))
            P.append(cur)
        return P

    def print_matrix(self, P):
        for p in P:
            print(p)
        print('')

    def publish_polygons(self, polygs, indexes, publisher):
        self.polygon_array_msg.header = polygs.header
        result = []
        for index in indexes:
            result.append(polygs.polygons[index])
        self.polygon_array_msg.polygons = result
        publisher.publish(self.polygon_array_msg)

    def low_polygon(self, positions):
        pos_y = [pos[1] for pos in positions]
        max_ = max(pos_y)
        return pos_y.index(max_)

    def get_polyg_points(self, planes):
        # print(planes)
        # a = Plane(Point3D(1, 1, 1), Point3D(2, 3, 4), Point3D(2, 2, 2))
        # a.normal_vector
        positions = []
        for polygon in planes:
            p = np.array(polygon).T
            positions.append([np.mean(p[0, :]), np.mean(p[1, :]), np.mean(p[2, :])])
        return positions

    def find_stair(self, positions, floor, planes, P):
        '''

        Args:
            positions: list of polygon median points
            floor: index of the floor

        Returns:

        '''

        # Find vertical sizes of planes (along Y-axis)
        sizes = {}
        for i, plane in enumerate(planes):
            miny = plane[0][1]
            maxy = plane[0][1]
            for point in plane:
                if miny > point[1]:
                    miny = point[1]
                if maxy < point[1]:
                    maxy = point[1]
            if maxy-miny < self.stair_limits['height'] and i != floor:
                sizes[i] = maxy-miny

        cands = sizes.keys()

        # Filter candidates
        # Check collinearity
        todelete = []
        for i in cands:
            for j in cands:

                if P[i][j] < 0.95:
                    todelete.append(j)
                # for i, p in enumerate(P):
                # print(i, p)
        cands = [cand for cand in cands if cand not in todelete]
        # Distances
        pos_z = {}
        pos_y = {}
        pos_x = {}
        for c in cands:
            pos_z[c] = positions[c][2]
            pos_y[c] = positions[c][1]
            pos_x[c] = positions[c][0]
        pos_z = sorted(pos_z.items(), key=lambda x: x[1])
        if len(cands) != 0:
            return cands, pos_z[0][0], pos_z[-1][0]
        else:
            return [], -1, -1

    def dist_points(self, P1, P2):
        sum = 0

        for comp1, comp2 in zip(P1, P2):
            sum += (comp1 - comp2) ** 2
        return sum ** .5
        # return ((P1[0]-P2[0])**2+(P1[1]-P2[1])**2+(P1[2]-P2[2])**2)**.5

    def distance_plane_point(self, S, P):
        A = abs(P[0]*S[0] + P[1]*S[1] + P[2]*S[2] + S[3])
        B = (S[0]**2 + S[1]**2 + S[2]**2)**.5
        return A / B

    def retrieve_stair_surface(self, polygons, ordered_planes):
        '''
        Assumption: (i) camera is vertical, not inclined, (ii) stair is vertical
        Args:
            polygons:
            ordered_planes:

        Returns:
            polygon: points that form stair plane polygon
            coefficientss: equation of corresponding surface
        '''
        # ERROR in m
        ERROR = 0.05
        polygon, coefficients = [], []
        planes = [np.array(p) for p in polygons]
        max_ = [min(plane, key=lambda x: x[1])for plane in planes]
        points = []
        two = [ordered_planes[-1], ordered_planes[0]]
        for index in two:
            max_point = max_[index]
            for point in planes[index]:
                if abs(max_point[1] - point[1]) < ERROR:
                    points.append(point)
        # p is a point-cloud centroid
        # n is normal
        p, n = self.plane_fit(points)
        d = -(p[0] * n[0] + p[1] * n[1] + p[2] * n[2])
        return points, n + [d]

    def plane_fit(self, points):
        """
        p, n = planeFit(points)

        Given an array, points, of shape (d,...)
        representing points in d-dimensional space,
        fit an d-dimensional plane to the points.
        Return a point, p, on the plane (the point-cloud centroid),
        and the normal, n.
        """
        from numpy.linalg import svd

        points = np.reshape(points, (np.shape(points)[0], -1)).T  # Collapse trialing dimensions
        assert points.shape[0] <= points.shape[1], "There are only {} points in {} dimensions.".format(points.shape[1],
                                                                                                       points.shape[0])
        ctr = points.mean(axis=1)
        x = points - ctr[:, np.newaxis]
        M = np.dot(x, x.T)  # Could also use np.cov(x) here.
        return ctr, svd(M)[0][:, -1]

    def list2polygon(self, points):
        '''
        Args:
            points: list of points
        Returns:
            polygon: ROS polygon message
        '''
        l = []
        for i in range(len(points)):
            self.point.x = points[i][0]
            self.point.y = points[i][1]
            self.point.z = points[i][2]
            l.append(self.point)
        return l

    def plane_polygon_msg_creator(self, header, points):
        p = PolygonStamped()
        p.header = header
        p.polygon.points = [Point32(x=point[0], y=point[1], z=point[2]) for point in points]
        return p

    def publish_stair_plane(self, points):
        msg = PolygonArray()
        header = Header()
        header.frame_id = "camera_depth_optical_frame"
        header.stamp = rospy.Time.now()
        msg.header = header
        msg.polygons = [self.plane_polygon_msg_creator(header, points)]
        msg.labels = [0]
        msg.likelihood = [np.random.ranf()]
        self.stair_plane_pub.publish(msg)

    def get_angle_planes(self, P1, P2):
        nom = abs(P1[0]*P2[0] + P1[1]*P2[1] + P1[2]*P2[2])
        return nom / ((P1[0]*P1[0] + P1[1]*P1[1] + P1[2]*P1[2])**.5 * (P2[0]*P2[0] + P2[1]*P2[1] + P2[2]*P2[2])**.5)

    def get_min_max_points(self, plane, func=min):
        ERROR = 0.05
        m_point = func(plane, key=lambda x: x[1])
        m_points = []
        for point in plane:
            if abs(m_point[1] - point[1]) < ERROR:
                m_points.append(point)
        return m_points

    def stair_parameters(self, planes, coefs, floor_index, surface_coeficients, indexes):
        h, p, n, angle = 0, 0, 0, 0
        angle = self.get_angle_planes(coefs[floor_index], surface_coeficients)
        n = len(indexes)
        for i, index in enumerate(indexes):
            points = self.get_min_max_points(planes[index], min)
            h_min = (sum([y[1] for y in points]))/len(points)
            points = self.get_min_max_points(planes[index], max)
            h_max = (sum([y[1] for y in points]))/len(points)
            h += abs(h_max - h_min)
            # index to the next surface
            if i != len(indexes)-1:
                p += self.distance_plane_point(coefs[indexes[i+1]], points[0])
        h /= len(indexes)
        p /= len(indexes) - 1
        return h, p, n, angle

    def publish_parameters(self, h, p, n, angle):
        array = [h, p, n, angle]
        array = Float32MultiArray(data=array)
        self.array_pub.publish(array)

    def broadcast_step_pose(self, position, orientation):
        self.br.sendTransform((position[0], position[1], position[2]),
                         tf.transformations.quaternion_from_euler(-orientation[0],  -orientation[1], -orientation[2]),
                         rospy.Time.now(),
                         "stair",
                         "camera_rgb_optical_frame")

    def get_bbox(self, polygons, positions, indexes, h, n):
        # 0th point is the farther point
        # -1th point is the closest point
        width = 0
        left = 100
        right = 0
        for point in polygons[indexes[0]]:
            if point[0] > right:
                right = point[0]
            if point[0] < left:
                left = point[0]
        width = abs(right - left)
        # Left down point
        p1_t = dcopy(positions[indexes[-1]])
        p1_t[0] -= width / 2
        p1_t[1] += h / 2
        # Right upper point
        p2_t = dcopy(positions[indexes[-1]])
        p2_t[0] += width / 2
        p2_t[1] -= h / 2
        p2_t[1] -= h * (n-1)
        p1 = self.project_point(p1_t)
        p2 = self.project_point(p2_t)
        return [p1[0], p1[1], p2[0], p2[1]]

    def project_point(self, point):
        z = point[2]
        width_z = 2 * z * np.tan(self.resolution['H']/2)
        width_z = 2 * z * np.tan(self.resolution['V'] / 2)
        w = self.resolution['width'] + int(point[0] / width_z * self.resolution['width'])
        h = self.resolution['height'] + int(point[1] / width_z * self.resolution['height'])
        return [w, h]

    def publish_bbox(self, bbox):
        array = Float32MultiArray(data=bbox)
        self.corner_pub.publish(array)

    def callback(self, coefs, polygs, cloud):
        '''
        Args:
            data: list of multiple plane coefficients
        '''
        # Retrieving planes and coefs
        planes = self.polygons2list(polygs)
        coefs = self.coefs2list(coefs)

        # Calculate perimeters
        # perimeters = self.get_perimeters(planes)

        # Calculate matrix of perpendicularity
        P = self.find_perpendicularity(coefs)

        # Calculate mean polygon centroids
        centroids = self.get_polyg_points(planes)
        if len(centroids) == 0:
            return None

        # I Find floor
        # Calculating the lowest polygon
        floor_index = self.low_polygon(centroids) # self.filter_floor_candidates(perimeters, candidates)

        # Publish floor polygon and its coefficients
        self.publish_polygons(polygs, [floor_index], self.floor_pub)

        # II Find stair
        # indexes contains plane indexes in the order of closeness
        indexes, closest, farther = self.find_stair(centroids, floor_index, planes, P)

        # Publish stair
        self.publish_polygons(polygs, indexes, self.stair_pub)

        # III Estimate stair surface
        ## Get upper points for closest and farther planes
        ## Fil in a surface
        if len(indexes) > 1:
            points, surface_coeficients = self.retrieve_stair_surface(planes, indexes)

            # Publish polygon
            self.publish_stair_plane(points)

            # IV Extract parameters
            h, p, n, angle = self.stair_parameters(planes, coefs, floor_index, surface_coeficients, indexes)
            self.publish_parameters(h, p, n, angle)

            # V Find first step and publish its pose
            self.broadcast_step_pose(centroids[indexes[-1]], coefs[indexes[-1]])

            # VI Retrieve corner points and publish a bounding box
            bbox = self.get_bbox(planes, centroids, indexes, h, n)
            if len(bbox) != 0:
                self.publish_bbox(bbox)





if __name__ == '__main__':
    DetectorDepth()

'''
        for topics in rospy.get_published_topics():
            for topic in topics:
                if '/multi_plane_estimate/output_coefficients' in topic:
                    print(topic)
'''
