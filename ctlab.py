#!/usr/bin/env python3

import pathlib
import os

import math as m
import numpy as np
from scipy import integrate

from attenuation_object import ObjectCollection, Rectangle, Circle
from misc import array_to_img
from projection import Projection




if __name__ == "__main__":
    # Make sure that the output/ directory exists, or create it otherwise.
    output_dir = pathlib.Path.cwd() / "output"
    if not output_dir.is_dir():
        output_dir.mkdir()

    ############################################ Sample, two squares
    # Create two-squares as objects
    print("Two-squares")

    rectangle_1 = Rectangle(-1, 0, 0, 1, 1)
    rectangle_2 = Rectangle(0, 1, -1, 0, 1)


    collection = ObjectCollection()
    collection.append(rectangle_1)
    collection.append(rectangle_2)

    ################## sample attenuation coefficient
    # Visualising miu(x,y), attenuation coefficient at every x and y-coordinate
    print("     Begin two-squares attenuation coefficient graph")
    array_to_img(
        collection.to_array(
            np.linspace(-2, 2, 100),
            np.linspace(-2, 2, 100),
        )).save(output_dir /"squares-original.png")

    # linspace are the x and y axis

    ################### sample projected coefficient

    print("     Begin two-squares projected coefficient graph")
    projection = Projection((0,np.pi), 50, (-2,2), 50)
    projection.add_object(rectangle_1, (0,10))
    projection.add_object(rectangle_2, (-10,0))

    # Problem: image is rotated to the right.
    array_to_img(projection.data).save(output_dir /"squares-pa.png")
    projection.back_project()

    array_to_img(projection.original_image).save(output_dir/"circle-bp.png")

    ############################################################### Circle
    print("Circle")

    circle_1 = Circle(0,0,1,1)
    circle_collection = ObjectCollection()
    circle_collection.append(circle_1)

    ################### Circle attenuation coefficient

    print("     Begin circle attenuation coefficient graph")
    array_to_img(circle_collection.to_array(
        np.linspace(-2, 2, 100),
        np.linspace(-2, 2, 100),
    )).save(output_dir / "circle-original.png")


'''    ################## Circle projected attenuation
    #circle_projection1 = circle_1.project_attenuation(0,1, eta_range)
    #print(f"circle proj 1: {circle_projection1}")

    print("     Begin circle projected coefficient graph")
    c_projection = Projection((0,np.pi), 50, (-2,2), 50)
    c_projection.add_object(circle_1,(-10,10))

    array_to_img(c_projection.data).save(output_dir /"circle-pa.png")

    c_projection.back_project()

    array_to_img(c_projection.original_image).save(output_dir/"circle-bp.png")'''
