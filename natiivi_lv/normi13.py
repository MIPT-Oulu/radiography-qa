# -*- coding: utf-8 -*-
"""
Created on Mon Sep 11 12:20:47 2023

@author: rytkysan
"""

import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import cv2
import pickle  # This is very handy for storing dictionaries
from pylinac.planar_imaging import HighContrastDiskROI, LowContrastDiskROI, PlanarResult
from pylinac.core.mtf import MTF, MomentMTF
from pylinac.core.geometry import Point
from pylinac.core.contrast import Contrast
from pylinac.core.image import DicomImage
from pylinac.core.utilities import ResultsDataMixin

from natiivi_lv.constants import (IMAGING_PARAMETERS, HIGH_CONTRAST_ROI_SETTINGS, LINE_PAIR_PATTERN_SETTINGS,
                                  LOW_CONTRAST_ROI_SETTINGS, UNIFORMITY_ROI_SETTINGS, TEST_ROI_SETTINGS,
                                  TEST_ROI_LPS_SETTINGS)
from natiivi_lv.utilities import ratio


class Normi13(ResultsDataMixin[PlanarResult]):
    """
    Class for performing image quality analysis of Normi-13 phantom images.

    Parameters
    ----------
    path : Path | str
        Path to the Dicom image to be analyzed.
    plot : bool, optional
        Set to true for saving result figures. The default is False.
    debug : bool, optional
        Set to True for plotting additional debugging figures. The default is False.
    fig_path : Path, optional
        Path for saving the result figures. The default is None.
    mtf_mode : str, optional
        Calculation method for the MTF curve.
        'relative' = Calculate a relative curve for each measured ROI.
        'moments' = Calculate a moments-based MTF for each measured ROI.
        See Handers et al 1997 and https://aapm.onlinelibrary.wiley.com/doi/epdf/10.1118/1.597928
        The default is 'relative'.
    high_contrast_threshold : float, optional
        Threshold for considering the ROI to be "seen". The default is 0.5.
    low_contrast_threshold : float, optional
        Threshold for considering the ROI to be "seen". The default is 0.05.
    visibility_threshold : float, optional
        Threshold for considering the ROI to be "seen". The default is 0.025.

    Returns
    -------
    None.

    """

    # DICOM tags for imaging parameters
    imaging_param = {}
    # Test name
    common_name = 'Normi-13'
    # Results
    results = {}
    # Analysis ROIs
    rois_orient_resize = {}
    # Averages and STDs from ROIs
    roi_avgs_stds = {}
    # High-contrast ROIs for MTF
    line_pair_rois = {}
    high_contrast_rois = {}
    low_contrast_rois = {}
    uniformity_rois = {}
    # Angle to correct phantom orientation
    phantom_angle: float
    # Side length of the phantom
    side_length: float
    # Center of the phantom
    phantom_center: tuple
    # Mean of uniformity ROIs
    bg_mean: float
    # Maximum deviation of uniformity ROIs
    uniformity_deviation: float

    # Crop parameters: the detected circles should be in the outer 25% of the image
    image_center = (0.2, 0.8)
    # Relative margin for cropping
    margin = 0.03

    def __init__(self,
                 path: Path | str,
                 plot: bool = False,
                 debug: bool = False,
                 fig_path: Path = None,
                 mtf_mode: str = 'relative',
                 high_contrast_threshold: float = 0.5,
                 low_contrast_threshold: float = 0.05,
                 visibility_threshold: float = 0.025):

        # Plot intermediate figures
        self.plot = plot
        # Plot more figures for debugging
        self.debug = debug
        # Path for saving figures
        self.fig_path = fig_path

        # Read data (12-bit grayscale image as 16-bit uint)
        self.dataset = DicomImage(path)
        # The phantom image should have high values for phantom targets
        # This is corrected also on the orientation function
        self.dataset.invert()
        # Convert negative values to positive
        self.dcm_img = abs(self.dataset.array)

        # 8-bit image for OpenCV algorithms
        self.img_8bit = (np.maximum(self.dcm_img, 0) / (self.dcm_img.max() + 1e-5) * 255).astype('uint8')

        # Imaging parameters
        for key, meta in IMAGING_PARAMETERS.items():
            try:  # Test which of the most relevant parameters are available
                self.imaging_param[key] = self.dataset.metadata[meta].value
            except KeyError:
                self.imaging_param[key] = 'None'

        # Hyperparameters for edge detection
        self.hough_hyperparam = {
            'method': cv2.HOUGH_GRADIENT,  # Detection method
            'dp': 4.0,  # Image res to accumulator resolution ratio 4.0
            'minDist': 500,  # Distance between circle centers 500
            'minRadius': 60,  # Circle radius 60
            'maxRadius': 80,  # 80
            'param1': 210,  # Image threshold 200
            'param2': 70,  # Accumulator threshold 50
        }

        # MTF analysis (relative or moments-based)
        choices = {'relative', 'moments'}
        if mtf_mode in choices:
            self.mtf_mode = mtf_mode
        else:
            self.mtf_mode = 'relative'

        # Thresholds
        self.high_contrast_threshold = high_contrast_threshold
        self.low_contrast_threshold = low_contrast_threshold
        self._low_contrast_method = Contrast.MICHELSON
        self.visibility_threshold = visibility_threshold

    def _crop_and_find_angle(self):
        """
        Detect phantom borders using Hough circles algorithm

        Raises
        ------
        ValueError
            Algorithm fails to detect the corner circles of the phantom.
        """

        # Detect circles on the image
        circles_det = cv2.HoughCircles(self.img_8bit.copy(), **self.hough_hyperparam)

        # Keep circles from the edges of the image.
        circles = []
        if circles_det is not None:
            circles_det = np.round(circles_det[0, :].astype('int'))  # Change circles_det type to integer.
            width, height = max(circles_det[:, 0]) - min(circles_det[:, 0]), max(circles_det[:, 1]) - min(
                circles_det[:, 1])
            # Check if items in circles_det array are on the edge area of image
            for i in range(np.shape(circles_det)[0]):
                if ((circles_det[i][0] < self.image_center[0] * width + min(circles_det[:, 0])
                     or circles_det[i][0] > self.image_center[1] * width + min(circles_det[:, 0]))
                        and (circles_det[i][1] < self.image_center[0] * height + min(circles_det[:, 1])
                             or circles_det[i][1] > self.image_center[1] * height + min(circles_det[:, 1]))):
                    circles.append(circles_det[i])

        # If under 4 circles_det were detected, then invert the image and
        # use Hough circles_det again and delete circles_det detected on the center area of image
        if len(circles) < 4:
            circles = []
            circles_det = cv2.HoughCircles(np.invert(self.img_8bit), **self.hough_hyperparam)

            if circles_det is not None:
                circles_det = np.round(circles_det[0, :].astype('int'))

                # Check if items in circles_det array are on the edge area of image
                for i in range(np.shape(circles_det)[0]):
                    if ((circles_det[i][0] < self.image_center[0] * int(self.dcm_img.shape[1])
                         or circles_det[i][0] > self.image_center[1] * int(self.dcm_img.shape[1]))
                            and (circles_det[i][1] < self.image_center[0] * int(self.dcm_img.shape[0])
                                 or circles_det[i][1] > self.image_center[1] * int(self.dcm_img.shape[0]))):
                        circles.append(circles_det[i])

        # No edge circles detected
        if len(circles) == 0:
            raise ValueError(f'Edges of the phantom not detected ({self.dataset.base_path}).')

        # Convert back to numpy array
        circles = np.array(circles)

        # Draw a circle and small rectangle on the places of detected circles.
        # Show the original dicom image and image where detected circles are marked with circle and small rectangle.
        if self.debug:
            # Show starting image
            fig, ax = plt.subplots(1, 2, figsize=(10, 5))
            ax[0].imshow(self.dcm_img, cmap='gray')
            ax[0].set_title(f'Starting image {self.dataset.base_path}')

            # Edge circles
            output = cv2.cvtColor(self.img_8bit, cv2.COLOR_GRAY2RGB)
            for (x, y, r) in circles:
                cv2.circle(output, (x, y), r, (255, 128, 0), 10)  # Draw circle
                cv2.rectangle(output, (x - 5, y - 5), (x + 5, y + 5), (0, 128, 255), 3)  # Draw circle center
            ax[1].imshow(output)
            ax[1].set_title('Edge circles')
            plt.tight_layout()
            self.fig_path.mkdir(exist_ok=True, parents=True)
            name = f'_{self.dataset.metadata.StationName}_{self.dataset.metadata.PatientID}_circles.png'
            spath = self.fig_path / (self.dataset.base_path.split('.')[0] + name)
            plt.savefig(spath)

            if self.debug:
                # Show previous figure
                plt.show()

                output = cv2.cvtColor(self.img_8bit, cv2.COLOR_GRAY2RGB)
                for (x, y, r) in circles_det:
                    cv2.circle(output, (x, y), r, (255, 128, 0), 10)  # Draw circle
                    cv2.rectangle(output, (x - 5, y - 5), (x + 5, y + 5), (0, 128, 255), 3)  # Draw circle center
                plt.axis('off')
                plt.imshow(output)
                plt.title('All detected circles')
                plt.show()

        # Find phantom center
        self.phantom_center = (int(np.average(circles[:, 0])), int(np.average(circles[:, 1])))

        # Phantom side length (also when rotated)
        # Sort by the sum of coordinates. This makes sure that the diagonal is not used.
        points = np.array([circles[i, :2] for i in np.argsort(np.sum(circles[:, :2], axis=1))])
        sides = [np.sqrt((points[0, 0] - points[1, 0]) ** 2 + (points[0, 1] - points[1, 1]) ** 2),
                 np.sqrt((points[0, 0] - points[2, 0]) ** 2 + (points[0, 1] - points[2, 1]) ** 2)]
        self.side_length = np.mean(sides)

        # Find phantom angle based on detected circles
        self._find_phantom_angle(circles)

        # Crop image based on found circle points
        xmin = min(circles[:, 0])
        xmax = max(circles[:, 0])
        ymin = min(circles[:, 1])
        ymax = max(circles[:, 1])

        # Add margins to the cropped image
        margin = self.side_length * self.margin

        # Values for cropping image, minimum of 0, maximum of image size
        ymin = int(max(ymin - margin, 0))
        ymax = int(min(ymax + margin, self.img_8bit.shape[0]))
        xmin = int(max(xmin - margin, 0))
        xmax = int(min(xmax + margin, self.img_8bit.shape[1]))

        # Crop using the obtained values
        self.dcm_img = self.dcm_img[ymin:ymax, xmin:xmax]
        self.img_8bit = self.img_8bit[ymin:ymax, xmin:xmax]

        # Correct center position to the cropped image
        self.phantom_center = (self.phantom_center[0] - xmin, self.phantom_center[1] - ymin)

        # Check the results
        if self.debug:
            plt.imshow(self.dcm_img, cmap='gray')
            plt.title('Cropped image')
            plt.show()

    def _find_phantom_angle(self, circles: np.ndarray):
        """
        Calculates the Normi-13 phantom orientation. Checks for the angle of the square phantom (including 90-degree
        rotations) and if the phantom is flipped.

        Parameters
        ----------
        circles : np.ndarray
            Array of edge circle coordinates.
        """

        # Empty image with points of the circles
        circle_image = np.zeros(self.img_8bit.shape).astype('uint8')

        # Draw lines between the detected points to get a continuous structure
        for ind, (x, y, _) in enumerate(circles):
            if ind == len(circles) - 1:
                point = (circles[0][0], circles[0][1])
            else:
                point = (circles[ind + 1][0], circles[ind + 1][1])
            cv2.line(circle_image, (x, y), point, color=(255,), thickness=1)

        # Binarize the image
        _, th1 = cv2.threshold(
            circle_image.copy(), np.min(circle_image), np.max(circle_image), cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Detect the contours of the drawn line shape (rectangle, hourglass, ...)
        contours, _ = cv2.findContours(th1.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        # Get the largest contour
        if type(contours) is tuple:
            contours = max(contours, key=cv2.contourArea)

        # Bounding rectangle with the smallest area
        angle = cv2.minAreaRect(contours.copy())[2]
        # Opencv returns angle between 0 and 90 degrees
        if angle > 45:
            angle = 90 - angle
        else:
            angle = -angle

        # Test for image flip: get measurements from test ROIs
        rotations = [0, 90, 180, 270]
        test_results = {
            key: [] for key in rotations
        }
        test_means = np.empty((4, 5))

        for rot_id, rot_90 in enumerate(rotations):
            for roi_id, (roi_name, stng) in enumerate(TEST_ROI_SETTINGS.items()):
                roi = HighContrastDiskROI(
                    self.dcm_img,
                    stng["angle"] - (angle + rot_90),
                    self.side_length * stng["roi radius"],
                    self.side_length * stng["distance from center"],
                    Point(self.phantom_center),  # Pattern center, reverse the x and y dimensions from OpenCV
                    self.low_contrast_threshold,
                )
                test_results[rot_90].append(roi)
                test_means[rot_id, roi_id] = roi.mean

        # Estimate the 90-degree rotation
        rotation_90: int

        # Correct orientation has lowest min value in the first roi and max value in second roi
        max_ind = np.unravel_index(test_means[:, :2].argmax(), test_means[:, :2].shape)
        min_ind = np.unravel_index(test_means[:, :2].argmin(), test_means[:, :2].shape)

        # Maximum @ Cu230 (1), minimum @ Cu000 (0)
        cnt_th = 0.05
        if max_ind[0] == min_ind[0] and max_ind[1] == 1 and min_ind[1] == 0:
            # Correct orientation, lps-roi (3) has higher absorption than opposite side (4) and center (2)
            if (ratio(test_means[max_ind[0], 3], test_means[max_ind[0], 2]) > cnt_th and
                    ratio(test_means[max_ind[0], 3], test_means[max_ind[0], 4]) > cnt_th):
                rotation_90 = rotations[max_ind[0]]

            # Inverted and flipped image (contrast rois correct, negative lps in opposite side)
            elif (ratio(test_means[max_ind[0], 4], test_means[max_ind[0], 2]) < -cnt_th and
                  ratio(test_means[max_ind[0], 4], test_means[max_ind[0], 3]) < -cnt_th):
                # Invert images
                self.dcm_img = self.dcm_img.max() - self.dcm_img + self.dcm_img.min()
                self.img_8bit = self.img_8bit.max() - self.img_8bit + self.img_8bit.min()

                # Flip along x-axis
                self.dcm_img = np.fliplr(self.dcm_img)
                self.img_8bit = np.fliplr(self.img_8bit)

                # After flip, correct the rotation 180 degrees
                if max_ind[0] in [0, 2]:
                    rotation_90 = rotations[max_ind[0]] - 180
                else:
                    rotation_90 = rotations[max_ind[0]]
            else:
                raise ValueError(f'Failed to detect orientation of the phantom ({self.dataset.base_path}).')

        # Maximum and minimum swapped, could be due to image flip or inverted image
        elif max_ind[0] == min_ind[0] and max_ind[1] == 0 and min_ind[1] == 1:
            # Flipped image (lps-roi in opposite side)
            if (ratio(test_means[max_ind[0], 4], test_means[max_ind[0], 2]) > cnt_th and
                    ratio(test_means[max_ind[0], 4], test_means[max_ind[0], 3]) > cnt_th):
                # Flip along x-axis
                self.dcm_img = np.fliplr(self.dcm_img)
                self.img_8bit = np.fliplr(self.img_8bit)

                # After flip, correct the rotation 180 degrees
                if max_ind[0] in [0, 2]:
                    rotation_90 = rotations[max_ind[0]] - 180
                else:
                    rotation_90 = rotations[max_ind[0]]

            # Inverted image (lps-roi in correct side)
            elif (ratio(test_means[max_ind[0], 3], test_means[max_ind[0], 2]) < -cnt_th and
                  ratio(test_means[max_ind[0], 3], test_means[max_ind[0], 4]) < -cnt_th):
                # Invert images
                self.dcm_img = self.dcm_img.max() - self.dcm_img + self.dcm_img.min()
                self.img_8bit = self.img_8bit.max() - self.img_8bit + self.img_8bit.min()

                rotation_90 = rotations[max_ind[0]]
            else:
                raise ValueError(f'Failed to detect orientation of the phantom ({self.dataset.base_path}).')
        else:
            raise ValueError(f'Failed to detect orientation of the phantom ({self.dataset.base_path}).')

        # Save the estimate of angle and 90-degree rotation
        self.phantom_angle = angle + rotation_90

        # Overlay test rois
        test_results = []
        for roi_name, stng in TEST_ROI_SETTINGS.items():
            if roi_name == 'lps_180':
                continue

            roi = HighContrastDiskROI(
                self.dcm_img,
                stng["angle"] - self.phantom_angle,
                self.side_length * stng["roi radius"],
                self.side_length * stng["distance from center"],
                Point(self.phantom_center),  # Pattern center, reverse the x and y dimensions from OpenCV
                self.low_contrast_threshold,
            )
            test_results.append(roi)
        if self.debug:
            fig, ax = plt.subplots(1)
            ax.imshow(self.dcm_img, 'gray')
            for roi in test_results:
                roi.plot2axes(ax, edgecolor='blue')
            ax.set_title('Test ROIs')
            plt.show()

    def _calculate_contrast_uniformity(self):
        """
        Calculates different contrast parameters for the 5 uniformity ROIs,
        7 linearity ROIs and 6 low-contrast ROIs.
        Overlays the ROI locations to the Dicom image if self.plot == True.
        
        "Returns"
        ------
        self.uniformity_rois
            Dictionary of HighContrastDiskROIs from 5 uniformity regions.
        self.high_contrast_rois
            Dictionary of HighContrastDiskROIs from 7 linearity regions.
        self.low_contrast_rois
            Dictionary of LowContrastDiskROIs from 6 low-contrast regions.
        self.bg_mean
            Average of the uniformity rois
        self.uniformity_deviation
            Maximum deviation of perimeter uniformity regions to the 
            center region (absolute value).
        """

        # Uniformity test
        for roi_name, stng in UNIFORMITY_ROI_SETTINGS.items():
            self.uniformity_rois[roi_name] = HighContrastDiskROI(
                self.dcm_img,
                stng["angle"] - self.phantom_angle,
                self.side_length * stng["roi radius"],
                self.side_length * stng["distance from center"],
                Point(self.phantom_center),  # Pattern center, reverse the x and y dimensions from OpenCV
                self.low_contrast_threshold,
            )

        # Background mean (average of the uniformity ROIs)
        uniformity_means = {key: self.uniformity_rois[key].mean for key in self.uniformity_rois}
        self.bg_mean = np.mean(list(uniformity_means.values()))

        # Maximum deviation from the center patch
        max_deviation = 0
        for key, u_mean in uniformity_means.items():
            dev = np.abs(u_mean - uniformity_means['u_center'])
            if key != 'u_center' and dev > max_deviation:
                max_deviation = dev
        self.uniformity_deviation = max_deviation

        # Contrast tests
        for roi_name, stng in HIGH_CONTRAST_ROI_SETTINGS.items():
            self.high_contrast_rois[roi_name] = HighContrastDiskROI(
                self.dcm_img,
                stng["angle"] - self.phantom_angle,
                self.side_length * stng["roi radius"],
                self.side_length * stng["distance from center"],
                Point(self.phantom_center),  # Pattern center, reverse the x and y dimensions from OpenCV
                self.low_contrast_threshold,
            )

        # Low contrast tests
        for roi_name, stng in LOW_CONTRAST_ROI_SETTINGS.items():
            self.low_contrast_rois[roi_name] = LowContrastDiskROI(
                self.dcm_img,
                stng["angle"] - self.phantom_angle,
                self.side_length * stng["roi radius"],
                self.side_length * stng["distance from center"],
                Point(self.phantom_center),  # Pattern center, reverse the x and y dimensions from OpenCV
                contrast_threshold=self.low_contrast_threshold,
                contrast_reference=self.bg_mean,
                contrast_method=self._low_contrast_method,
                visibility_threshold=self.visibility_threshold,
            )

        # Show ROIs on the phantom image
        if self.plot:
            fig, ax = plt.subplots(1, 2, figsize=(10, 5))
            ax[0].imshow(self.dcm_img, 'gray')
            for roi in self.uniformity_rois.values():
                roi.plot2axes(ax[0], edgecolor='blue')
            ax[0].set_title('Uniformity ROIs')

            ax[1].imshow(self.dcm_img, 'gray')
            for roi in self.high_contrast_rois.values():
                roi.plot2axes(ax[1], edgecolor='blue')
            for roi in self.low_contrast_rois.values():
                roi.plot2axes(ax[1], edgecolor='blue')
            ax[1].set_title('High- and low-contrast ROIs')
            plt.tight_layout()
            name = (f'_{self.dataset.metadata.StationName}_{self.dataset.metadata.PatientID}_'
                    f'{self.dataset.metadata.SeriesDate}_{self.dataset.metadata.SeriesTime[:-7]}_contrast_rois.png')
            spath = self.fig_path / (self.dataset.base_path.split('.')[0] + name)
            plt.savefig(spath)
            # Show figure only on debug mode
            if self.debug:
                plt.show()

    def _estimate_mtf(self):
        """
        Detects the orientation and center of the line pair element.
        Calculates Modulation transfer function based on the line pair regions.
        
        Raises
        ------
        NotImplementedError: 'relative' or 'moments' should be used as
        MTF algorithms (self.mtf_mode).
        
        "Returns"
        ------
        self.mtf
            The Modulation transfer function.
        """

        # Find the Line pair square element and its orientation

        # Crop the dicom image to include only the line pair element
        x = (int(self.high_contrast_rois['lps'].center.x - self.high_contrast_rois['lps'].diameter),
             int(self.high_contrast_rois['lps'].center.x + self.high_contrast_rois['lps'].diameter))
        y = (int(self.high_contrast_rois['lps'].center.y - self.high_contrast_rois['lps'].diameter),
             int(self.high_contrast_rois['lps'].center.y + self.high_contrast_rois['lps'].diameter))
        lps_cropped = self.img_8bit[y[0]:y[1], x[0]:x[1]]
        lps_cropped_dcm = self.dcm_img[y[0]:y[1], x[0]:x[1]]

        # Find the line pair pattern using adaptive thresholding (less false positives from test lines than with Otsu)
        th1 = cv2.adaptiveThreshold(
            cv2.medianBlur(lps_cropped.copy(), 5), np.max(lps_cropped), cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 9, -1)  # Block size, subtraction constant

        # Detect the square element
        contours, _ = cv2.findContours(th1.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        # The largest contour should include the square element
        largest = max(contours, key=cv2.contourArea)

        # Calculate the smallest area rectangle from the largest contour
        # Returns center coordinates, width, height, angle of rotation
        rect = cv2.minAreaRect(largest.copy())

        # Draw the detected test pattern (only if self.debug == True)
        if self.debug:
            output = cv2.cvtColor(lps_cropped, cv2.COLOR_GRAY2RGB)
            cv2.drawContours(output, largest, -1, (0, 255, 0), 3)
            plt.imshow(output)
            plt.show()

        # Find test pattern orientation
        test_patterns = []
        for roi_name, stng in TEST_ROI_LPS_SETTINGS.items():
            roi = HighContrastDiskROI(
                lps_cropped_dcm,
                stng['angle'] - self.phantom_angle,
                np.mean(rect[1]) * stng['roi radius'],
                np.mean(rect[1]) * stng['distance from center'],
                Point(rect[0][0], rect[0][1]),  # Pattern center, reverse the x and y dimensions from OpenCV
                self.high_contrast_threshold,
            )
            test_patterns.append(roi.std)

        # Location of the largest line pair pattern
        lps_angle = [90, 0, 270, 180]
        lps_angle = lps_angle[np.argmax(test_patterns)]
        # The ROIs are not defined perfectly from the center
        roi_offsets_x = [0, 0, 0, 0]
        roi_offsets_y = [0, 0, 0, 0]
        roi_offsets_x = roi_offsets_x[np.argmax(test_patterns)]
        roi_offsets_y = roi_offsets_y[np.argmax(test_patterns)]

        # Get ROI measurements from the line pair patterns

        # High contrast ROIs from the test image
        for roi_name, stng in LINE_PAIR_PATTERN_SETTINGS.items():
            self.line_pair_rois[roi_name] = HighContrastDiskROI(
                lps_cropped_dcm,
                stng['angle'] - self.phantom_angle - lps_angle,
                np.mean(rect[1]) * stng['roi radius'],
                np.mean(rect[1]) * stng['distance from center'],
                Point(rect[0][0] + roi_offsets_x, rect[0][1] + roi_offsets_y),
                # Pattern center, reverse the x and y dimensions from OpenCV
                self.high_contrast_threshold,
            )

        # ROI spacings
        spacings = [roi['lp/mm'] for roi in LINE_PAIR_PATTERN_SETTINGS.values()]
        # Relative MTF
        if self.mtf_mode == 'relative':
            self.mtf = MTF.from_high_contrast_diskset(
                diskset=list(self.line_pair_rois.values()), spacings=spacings
            )
        # Moments-based MTF
        elif self.mtf_mode == 'moments':
            self.mtf = MomentMTF.from_high_contrast_diskset(
                diskset=list(self.line_pair_rois.values()), lpmms=spacings
            )
            self.mtf.spacings = spacings
        else:
            raise NotImplementedError(f'The option {self.mtf_mode} for calculating MTF is not implemented.')

        # Show ROIs and the MTF curve
        if self.plot:
            fig, ax = plt.subplots(1, 2, figsize=(10, 5))
            ax[0].imshow(lps_cropped_dcm, 'gray')
            for roi in self.line_pair_rois.values():
                roi.plot2axes(ax[0], edgecolor='blue')
            ax[0].set_title('Linepair ROIs')
            self.mtf.plot(ax[1])
            plt.tight_layout()
            name = (f'_{self.dataset.metadata.StationName}_{self.dataset.metadata.PatientID}'
                    f'_{self.dataset.metadata.SeriesDate}_{self.dataset.metadata.SeriesTime[:-7]}_mtf.png')
            spath = self.fig_path / (self.dataset.base_path.split('.')[0] + name)
            plt.savefig(spath)
            # Show figure only on debug mode
            if self.debug:
                plt.show()
            plt.close()

    def save_results(self, save_path: Path):
        """
        Saves Normi-13 analysis results to a pickle file.

        Parameters
        ----------
        save_path : Path
            Path to the results file.    
        """

        # Save results
        self.results['imaging_params'] = self.imaging_param

        # Save contrast results
        if self.high_contrast_rois:
            for roi_name, roi in self.high_contrast_rois.items():
                if roi_name != 'lps':
                    # Save to final results
                    name_mean = roi_name + '_mean'
                    self.results[name_mean] = roi.mean

                    name_std = roi_name + '_std'
                    self.results[name_std] = roi.std

        # Institution and station name for results folder
        dir_name = self.imaging_param['institution_name'] + ' ' + self.imaging_param['station_name']
        # replace possible occurrences of / from the name
        dir_name = dir_name.replace('/', '')

        # Save results using measurement date and increasing number

        # Make directories:
        save_path = save_path / dir_name
        save_path.mkdir(exist_ok=True, parents=True)

        # Save
        series_time = self.dataset.metadata.SeriesTime.split('.')[0]
        save_path = save_path / f'{self.dataset.metadata.SeriesDate}_{series_time}'

        # Save to a Pickle file
        with open(save_path, 'wb') as output:
            pickle.dump(self.results, output, pickle.HIGHEST_PROTOCOL)

    def _generate_results_data(self) -> PlanarResult:
        """
        Generates the main results of the Normi-13 analysis. This function is
        not called directly, instead self.results_data() can be called.

        Returns
        -------
        PlanarResult
            A results object with most important contrast and mtf parameters.
            More detailed results could be obtained outside the results_data 
            function using Normi13-class attributes.

        """

        area = self.side_length ** 2 / self.dataset.dpmm ** 2 if self.dataset.dpmm is not None else 0
        data = PlanarResult(
            analysis_type=self.common_name,
            median_contrast=np.median([roi.contrast for roi in self.low_contrast_rois.values()]),
            median_cnr=np.median(
                [roi.contrast_to_noise for roi in self.low_contrast_rois.values()]
            ),
            num_contrast_rois_seen=sum(
                roi.passed_visibility for roi in self.low_contrast_rois.values()
            ),
            phantom_center_x_y=(self.phantom_center[0], self.phantom_center[1]),
            low_contrast_rois=[roi.as_dict() for roi in self.low_contrast_rois.values()],
            percent_integral_uniformity=None,
            phantom_area=area,
        )

        if self.mtf is not None:
            data.mtf_lp_mm = {
                p: self.mtf.relative_resolution(p) for p in range(10, 91, 10)
            }
        return data

    def analyze(self):
        """
        A simple main function to run the analysis pipeline.
        """

        # Crop the image based on circles at the edge of the phantom,
        self._crop_and_find_angle()

        # Calculate contrast and uniformity from the phantom image
        self._calculate_contrast_uniformity()

        # Calculate MTF
        self._estimate_mtf()
