from typing import List
import numpy as np
from math import sqrt
from numba import jit

class Object:
    def __init__(self, center: np.array, vertices: List[np.array]):
        #self.heading = heading
        self.position_center = center
        self.vertices = vertices
        # TODO: an object should be able to hold more than 4 sides
        #self.sides = [[vertices[0], vertices[1]],
        #              [vertices[1], vertices[2]],
        #              [vertices[2], vertices[3]],
        #              [vertices[3], vertices[0]]]
        self.sides = self.vertices_to_sides(vertices)
        self.lines = self.eval_lines(self.sides)
        self.radius = self.eval_radius()

    def vertices_to_sides(self, vertices):
        sides = [[vertices[-1], vertices[0]]]
        for i in range(len(vertices)-1):
            sides.append([vertices[i], vertices[i+1]])
        return sides
   
    """Evaluate radius of circle surrounding vehicle"""
    def eval_radius(self) -> float:
        # given the initial shape of the vehicle, that for now we consider as rectangular, we evaluate the radius.
        radius = self.dist_point_point(self.position_center, self.vertices[0])
        return radius

    """Evaluate lines passing throught vertices"""
    def eval_lines(self, sides) -> List[np.array]:
        # evaluate the lines passing through the vertices and insert them in a list. every line is expressed as an
        # array with 3 values [a,b,c]
        lines = []
        for side in self.sides:
            lines.append(self.eval_line_point_point(side[0], side[1]))
        return lines

    # Evaluate line passing throught two points"""
    def eval_line_point_point(self, point1: np.array, point2: np.array) -> np.array:
        # evaluate the line passing throught two points, expressing it as a vector with 3 values [a,b,c]
        delta_x = point1[0] - point2[0]
        if delta_x != 0: # Check if the line is vertical, otherwise there is a division by 0
            delta_y = point1[1] - point2[1]
            m = delta_y / delta_x
            c = (delta_x * point1[1] - delta_y * point1[0]) / delta_x
            line = np.array([m, -1, c])
        else:
            line = np.array([1, 0, - point1[0]]) # The equation of a vertical line
        return line

    """ Methods useful for evaluation distances"""
    # METHOD 1: The distance between two points
    @staticmethod
    @jit(nopython=True)
    def dist_point_point(point1: np.array, point2: np.array) -> float:
        distance = sqrt((point1[0]-point2[0])*(point1[0]-point2[0]) + (point1[1]-point2[1])*(point1[1]-point2[1]))
        return distance

    # METHOD 2: The distance between a line (defined in the normal form $a x+b y + c = 0$) and a point
    def dist_point_line(self, line: np.array, point: np.array) -> float:
        distance = abs(line[0]*point[0] + line[1]*point[1] + line[2]) / sqrt(line[0]*line[0] + line[1]*line[1])
        return distance

    # METHOD 3: find the closest point P on a given line $a x+b y + c = 0$ (which is the projection)
    # to a given point A ($x_0,y_0$)
    def find_point_projection(self,line: np.array, point: np.array) -> np.array:
        # One formula for doing it
        a = line[0]
        b = line[1]
        c = line[2]
        x = b / a * (a * a * point[1] + a * b * point[0] - b * c) / (b * b + a * a) - c / a
        y = (a * a * point[1] + a * b * point[0] - b * c) / (b * b + a * a)
        projection = np.array(x, y)
        return projection

    # METHOD 4: find if a point P($x_0,y_0$) on a line stands between two other points A($x_1,y_1$)
    # and B($x_2,y_2$) on  same line
    def is_point_in_segment(self, pointP: np.array, pointA: np.array, pointB: np.array) -> bool:
        """
        PSEUDOCODE

        IF distance AP + distance BP == distance AB
            P is between A and B in the same line
        """
        return self.dist_point_point(pointA,pointP) + self.dist_point_point(pointB,pointP) == self.dist_point_point(pointA,pointB)

    # METHOD 5: find if a point H has its projection between two other points A and B
    def is_point_in_segment_shaodow(self, pointH: np.array, pointA: np.array, pointB: np.array, line: np.array) -> bool:
        """
        PSEUDOCODE

        Find point P, which is projection of H on the line passing on A and B with METHOD 3
        IF P is between AB (METHOD 4)
            H is in the "shadow" of segment AB, return True
        ELSE
            H is NOT in the "shadow" of segment AB, return False
        """

        pointP = self.find_point_projection(line, pointH)

        if self.is_point_in_segment(pointP, pointA, pointB):
            return True
        else:
            return False


    # METHOD 6: distance of a point to a segment
    def dist_point_segment(self, pointH: np.array, pointA: np.array, pointB: np.array, line: np.array) -> float:
        """
        PSEUDOCODE

        IF point H is in the "shadow" of segment AB (METHOD 5)
            Find distance dist , line to point (METHOD 2)

        ELSE
            - find the distance between the points H and A (METHOD 1)
            - find the distance between the points H and B (METHOD 1)
            - keep the smallest distance dist.
        return dist
        """

        if self.is_point_in_segment_shaodow(pointH, pointA, pointB, line):
            distance = self.dist_point_line(line, pointH)
        else:
            distance = min(self.dist_point_point(pointH, pointA), self.dist_point_point(pointH, pointB))

        return distance

    # METHOD 7: distance of a point to an Obj
    def dist_point_object(self, pointH: np.array, obj) -> float:
        """
        PSEUDOCODE

        FOR every side of the polygon O
            Apply METHOD 6
        Keep the smallest distance dist
        return dist
        """
        points_obj = obj.vertices
        lines_obj = obj.lines
        for lin in obj.lines:
            self.dist_point_segment(pointH,) # IT HAS TO BE COMPLETED
        distance = 5
        return distance

    # This is the far-sighted distance, which is approximation of the distance. Very fast to evaluate
    def FSdist(self, radius1: float, radius2: float, center1: np.array, center2: np.array) -> float:
        distance = self.dist_point_point(center1, center2) - radius1 - radius2
        return distance

    # This is the near-sighted distance, which is the precise distance.
    def NSdist(self, obj) -> float:
        """
        PSEUDOCODE

        FOR every vertex of V
            apply METHOD 7 and keep the smallest distance dist1
        FOR every vertex of O
            apply METHOD 7 and keep the smallest distance dist2
        return the smallest distance found between dist1 and dist2

        """
        distance = 5
        return distance

