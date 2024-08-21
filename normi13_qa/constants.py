# Settings for uniformity and contrast rois a
UNIFORMITY_ROI_SETTINGS = {
    'u_center': {'distance from center': 0, 'angle': 0, 'roi radius': 0.1},
    'u_tl': {'distance from center': 0.534, 'angle': -135, 'roi radius': 0.03},
    'u_tr': {'distance from center': 0.534, 'angle': -45, 'roi radius': 0.03},
    'u_br': {'distance from center': 0.534, 'angle': 45, 'roi radius': 0.03},
    'u_bl': {'distance from center': 0.534, 'angle': 135, 'roi radius': 0.03},
}

LOW_CONTRAST_ROI_SETTINGS = {
    'lc_08': {'distance from center': 0.312, 'angle': -35.5, 'roi radius': 0.013},
    'lc_12': {'distance from center': 0.275, 'angle': -23.5, 'roi radius': 0.013},
    'lc_20': {'distance from center': 0.255, 'angle': -7.5, 'roi radius': 0.013},
    'lc_28': {'distance from center': 0.255, 'angle': 7.5, 'roi radius': 0.013},
    'lc_40': {'distance from center': 0.275, 'angle': 23.5, 'roi radius': 0.013},
    'lc_56': {'distance from center': 0.312, 'angle': 35.5, 'roi radius': 0.013},
}

LOW_CONTRAST_BG_ROI_SETTINGS = {
    'lc_08': {'distance from center': 0.259, 'angle': -43.5, 'roi radius': 0.02},
    'lc_12': {'distance from center': 0.215, 'angle': -30, 'roi radius': 0.02},
    'lc_20': {'distance from center': 0.189, 'angle': -10, 'roi radius': 0.02},
    'lc_28': {'distance from center': 0.189, 'angle': 10, 'roi radius': 0.02},
    'lc_40': {'distance from center': 0.215, 'angle': 30, 'roi radius': 0.02},
    'lc_56': {'distance from center': 0.259, 'angle': 43.5, 'roi radius': 0.02},
}

# High contrast ROIs and the location of the line pair pattern
HIGH_CONTRAST_ROI_SETTINGS = {
    'cu_000': {'distance from center': 0.329, 'angle': -139, 'roi radius': 0.02},
    'cu_030': {'distance from center': 0.287, 'angle': -150, 'roi radius': 0.02},
    'cu_065': {'distance from center': 0.258, 'angle': -164, 'roi radius': 0.02},
    'cu_100': {'distance from center': 0.2488, 'angle': 180, 'roi radius': 0.02},
    'cu_140': {'distance from center': 0.258, 'angle': 164, 'roi radius': 0.02},
    'cu_185': {'distance from center': 0.287, 'angle': 150, 'roi radius': 0.02},
    'cu_230': {'distance from center': 0.329, 'angle': 139, 'roi radius': 0.02},
    'lps': {'distance from center': 0.315, 'angle': 90, 'roi radius': 0.08},
}

# High contrast ROIs and the location of the line pair pattern
TEST_ROI_SETTINGS = {
    'cu_000': {'distance from center': 0.329, 'angle': -139, 'roi radius': 0.02},
    'cu_230': {'distance from center': 0.329, 'angle': 139, 'roi radius': 0.02},
    'u_center': {'distance from center': 0, 'angle': 0, 'roi radius': 0.1},
    'lps': {'distance from center': 0.315, 'angle': 90, 'roi radius': 0.08},
    'lps_180': {'distance from center': 0.315, 'angle': -90, 'roi radius': 0.08},
}

# High contrast ROIs and the location of the line pair pattern
TEST_ROI_LPS_SETTINGS = {
    '0': {'distance from center': 0.25, 'angle': 0, 'roi radius': 0.15},
    '90': {'distance from center': 0.25, 'angle': 90, 'roi radius': 0.15},
    '180': {'distance from center': 0.25, 'angle': 180, 'roi radius': 0.15},
    '270': {'distance from center': 0.25, 'angle': 270, 'roi radius': 0.15},
}

# Line pair pattern settings
LINE_PAIR_PATTERN_SETTINGS = {
    'roi 1': {
        'distance from center': 0.385,  # 200 pixels, 530 pattern side length
        'angle': 74,
        'roi radius': 0.04,
        'lp/mm': 0.6,
    },
    'roi 2': {
        'distance from center': 0.292,
        'angle': 84.5,
        'roi radius': 0.034,
        'lp/mm': 0.7,
    },
    'roi 3': {
        'distance from center': 0.225,
        'angle': 101,
        'roi radius': 0.032,
        'lp/mm': 0.8,
    },
    'roi 4': {
        'distance from center': 0.183,
        'angle': 128,
        'roi radius': 0.028,
        'lp/mm': 0.9,
    },
    'roi 5': {
        'distance from center': 0.204,
        'angle': 160,
        'roi radius': 0.026,
        'lp/mm': 1,
    },
    'roi 6': {
        'distance from center': 0.2547,
        'angle': 179,
        'roi radius': 0.024,
        'lp/mm': 1.2,
    },
    'roi 7': {
        'distance from center': 0.313,
        'angle': -171,
        'roi radius': 0.018,
        'lp/mm': 1.4,
    },
    'roi 8': {
        'distance from center': 0.366,
        'angle': -165,
        'roi radius': 0.016,
        'lp/mm': 1.6,
    },
    'roi 9': {
        'distance from center': 0.408,
        'angle': 22,
        'roi radius': 0.012,
        'lp/mm': 1.8,
    },
    'roi 10': {
        'distance from center': 0.345,
        'angle': 17,
        'roi radius': 0.010,
        'lp/mm': 2.0,
    },
    'roi 11': {
        'distance from center': 0.283,
        'angle': 11,
        'roi radius': 0.009,
        'lp/mm': 2.2,
    },
    'roi 12': {
        'distance from center': 0.228,
        'angle': 0,
        'roi radius': 0.008,
        'lp/mm': 2.5,
    },
    'roi 13': {
        'distance from center': 0.168,
        'angle': -30,
        'roi radius': 0.007,
        'lp/mm': 2.8,
    },
    'roi 14': {
        'distance from center': 0.160,
        'angle': -48,
        'roi radius': 0.007,
        'lp/mm': 3.1,
    },
    'roi 15': {
        'distance from center': 0.174,
        'angle': -67,
        'roi radius': 0.007,
        'lp/mm': 3.4,
    },
    'roi 16': {
        'distance from center': 0.200,
        'angle': -82,
        'roi radius': 0.007,
        'lp/mm': 3.7,
    },
    'roi 17': {
        'distance from center': 0.270,
        'angle': -99,
        'roi radius': 0.006,
        'lp/mm': 4.0,
    },
    'roi 18': {
        'distance from center': 0.313,
        'angle': -105,
        'roi radius': 0.004,
        'lp/mm': 4.3,
    },
    'roi 19': {
        'distance from center': 0.362,
        'angle': -109,
        'roi radius': 0.003,
        'lp/mm': 4.6,
    },
    'roi 20': {
        'distance from center': 0.411,
        'angle': -113,
        'roi radius': 0.002,
        'lp/mm': 5.0,
    },
}

# Imaging parameters relevant to CR phantom images
IMAGING_PARAMETERS = {'acquisition_date': 0x00080022,
                      'modality': 0x00080060,
                      'manufacturer': 0x00080070,
                      'patient_ID': 0x00100020,
                      'kVp_setting': 0x00180060,
                      'exposure': 0x00181152,
                      'exposure_uAs': 0x00181153,
                      'measurement_date': 0x00080022,
                      'tube_current': 0x00181151,
                      'protocol_name': 0x00181030,
                      'institution_name': 0x00080080,
                      'exposure_time': 0x00181150,
                      'filter_type': 0x00181160,
                      'grid': 0x00181166,
                      'distance_source_to_detector': 0x00181110,
                      'model_name': 0x00081090,
                      'station_name': 0x00081010,
                      'focal_spot': 0x00181190}