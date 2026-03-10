import numpy as np
# CAN_DIAB_TY  (FORMAT: DIABTY)
DIABTY_MAP = {
    1:   "No",
    2:   "Type I",
    3:   "Type II",
    4:   "Type Other",
    5:   "Unknown",
    998: "Diabetes Status Unknown",
    -9999: "MISSING",
}

# CAN_DIAL  (FORMAT: DIALTYLI)
DIALTYLI_MAP = {
    1:   "No dialysis",
    2:   "Hemodialysis",
    3:   "Peritoneal Dialysis",
    4:   "CAVH (Continuous Arteriovenous Hemofiltration)",
    5:   "CVVH (Continuous Venous/Venous Hemofiltration)",
    998: "Unknown",
    999: "Dialysis-Unknown Type Performed",
    -9999: "MISSING",
}

# CAN_MED_COND & REC_MED_COND  (FORMAT: MEDCOND)
MEDCOND_MAP = {
    1: "In Intensive Care Unit",
    2: "Hospitalized (not ICU)",
    3: "Not Hospitalized",
    -9999: "MISSING",
}

# CAN_PEPTIC_ULCER  (FORMAT: PEPULCER)
PEPULCER_MAP = {
    1:   "No",
    2:   "Yes, active within the last year",
    3:   "Yes, not active within the last year",
    4:   "Yes, activity unknown",
    998: "Unknown",
    -9999: "MISSING",
}

# DON_HIST_DIAB  (FORMAT: HSTDBDR)
HSTDBDR_MAP = {
    1:   "No",
    2:   "Yes, 0–5 years",
    3:   "Yes, 6–10 years",
    4:   "Yes, >10 years",
    5:   "Yes, duration unknown",
    998: "Unknown",
    -9999: "MISSING",
}

# REC_ACUTE_REJ_EPISODE  (FORMAT: ACREJEP)
ACREJEP_MAP = {
    1: "Yes, treated with anti-rejection agent",
    2: "Yes, none treated with additional agent",
    3: "No",
    -9999: "MISSING",
}

# REC_FAIL_CAUSE_TY (Format: GRFFL)
GRFFL_MAP = {
    **{code: "Rejection"           for code in [101,102,110]},
    **{code: "Primary failure"     for code in [103,109,112]},
    **{code: "Thrombosis"           for code in [104]},
    **{code: "Infection"           for code in [105,111]},
    **{code: "Complications/Recurrent Disease" for code in [106,107,108]},
    **{code: "Not Kidney" for code in list(range(200, 500))},
    999: "OTHER, SPECIFY",
    -9999: "MISSING",
}

# TFL_ACUTE_REJ_BIOPSY_CONFIRMED  (FORMAT: BPSCNFMF)
BPSCNFMF_MAP = {
    1:   "Biopsy not done",
    2:   "Yes, rejection confirmed",
    3:   "Yes, rejection not confirmed",
    998: "Unknown",
    -9999: "MISSING",
}

# REC_TX_TY  (FORMAT: TXTYPE)
TXTYPE_MAP = {
    1: "Single donor, single organ type",
    2: "Single donor, multiple organ types",
    3: "Multiple donors, single organ type",
    4: "Multiple donors, multiple organ types",
    -9999: "MISSING",
}

# TFL_DISEASE_RECUR (Format: DISRECR)
DISRECR_MAP = {
    1: "No recurrence",
    2: "Suspected recurrFence (not confirmed or unknown if confirmed by biopsy)",
    3: "Biopsy confirmed recurrence",
    998: "Unknown",
    -9999: "MISSING",
}

FUNCSTAT_MAP = {
    1: "No Assistance",
    2: "Some Assistance",
    3: "Total Assistance",
    # Pediatric 30% & below -> "Totally dependent"
    **{code: "Totally dependent" for code in [2010,2020,4010,4020,4030]},
    **{2030: "Totally dependent"},
    # Pediatric 40%–60% -> "Partially dependent"
    **{code: "Partially dependent" for code in [4040,4050,4060]},
    **{2040: "Partially dependent", 2050: "Partially dependent", 2060: "Partially dependent"},
    # Pediatric 70%–90% -> "Independent with effort"
    **{code: "Independent with effort" for code in [4070,4080,4090]},
    **{2070: "Independent with effort", 2080: "Independent with effort", 2090: "Independent with effort"},
    # 100% -> "Fully independent"
    **{4100: "Fully independent", 2100: "Fully independent"},
    # Special codes
    996: "Not applicable",
    998: "Unknown",
    -9999: "MISSING",
}

CANDSTAT_MAP = {
    **{code: "Status 1" for code in [1010,1020,1090,1110,2010,2020,2090,2110,3010,6010,6011,6012,9010]},
    **{code: "Status 2" for code in [1030,1120,2030,2120,6002,6020,6030,9020]},
    **{code: "Status 3-6" for code in [1130,1140,1150,1160,2130,2140,2150,2160,6004,6040]},
    **{code: "Temporarily Inactive" for code in [1999,2999,3999,4099,4999,5099,5999,6999,7999,8099,8999,9099,9999]},
    **{code: "Active" for code in [4010,4020,4050,4060,5010,7010,8010,9030]},
    **{code: "Other" for code in [3020,4998,6025,6029] + list(range(6050,6300))},
    **{code: "MISSING" for code in [0,-9999]},
}

RACE_MAP = {
    8:    "White",       16:   "Black",
    32:   "Native American", 64:   "Asian",
    128:  "Pacific Islander", 256:  "Arab/Middle Eastern",
    512:  "Indian Subcontinent", 1024: "Unknown",
    2000: "Hispanic/Latino", 
    **{code: "Multiracial" for code in list(range(1,8)) + list(range(9,16)) + list(range(17,32)) + list(range(33,64)) + list(range(65,128)) + list(range(129,256)) + list(range(257,512)) + list(range(513,1024)) + list(range(1025,2000)) + list(range(2001,5000))},
    -9999: "MISSING",
}

REMCD_MAP = {
    **{code: "Transplanted" for code in [4,15]},
    **{code: "Died"          for code in [8,21,23]},
    **{code: "Unsuitable"    for code in [5,13]},
    **{code: "Transferred"   for code in [6,7,14,22]},
    **{code: "Admin/Error"   for code in [10,11,16,24]},
    **{code: "Inactive"      for code in [12,20]},
    **{code: "Other"      for code in [9,17,18,19]},
    -9999: "MISSING",
}

INACTRSN_MAP = {
    **{code: "Cannot contact" for code in [1]},
    **{code: "Patient choice" for code in [2]},
    **{code: "Work‐up incomplete" for code in [3]},
    **{code: "Insurance issues" for code in [4]},
    **{code: "Non‐compliance" for code in [5]},
    **{code: "Substance use" for code in [6]},
    **{code: "Too sick"       for code in [7]},
    **{code: "Too well"       for code in [8]},
    **{code: "Weight issue"   for code in [9]},
    **{code: "VAD/inactivation" for code in [10,11,12]},
    **{code: "Surgeon unavailable" for code in [13]},
    **{code: "Living‐donor only"   for code in [14]},
    **{code: "Admin adjustment"    for code in [15]},
    -9999: "MISSING",
}

TXPROC_MAP = {
    101: "Left Kidney",
    102: "Right Kidney",
    103: "EN_BLOC",
    104: "Sequential Kidney",
    105: "Hemi-Rengal",
    **{code: "Other"      for code in list(range(200,800))},
    -9999: "MISSING",
}

# Mappings for high cardinality features: Top-10 uniques are extracted

# format: DGN
DGN_MAP = {
    **{code: "Kidney Glomerulonephritis"
       for code in [3000,3001,3002,3003,3004,3005,3006,3041,3043]},
    **{code: "Kidney Tubulointerstitial disease"
       for code in [3027,3028,3044,3045,3046,3047,3048,3049,3063]},
    **{code: "Cystic or hereditary kidney disease"
       for code in [3008,3010,3014,3015,3024,3029,3035,3064,3065]},
    **{code: "Kidney Diabetic or metabolic nephropathy"
       for code in [3009,3011,3012,3013,3038,3039,3069,3070,3071]},
    **{code: "Kidney Hypertensive or vascular injury"
       for code in [3026,3034,3040,3050,3066]},
    **{code: "Kidney Obstructive or reflux nephropathy"
       for code in [3007,3030,3032,3036,3042,3052]},
    **{code: "Kidney Congenital structural anomaly"
       for code in [3025,3031,3061]},
    3037: "Kidney Retransplant or graft failure",
    **{code: "Kidney Renal neoplasm"
       for code in [3020,3021,3022,3023,3058]},
    **{code: "Kidney Autoimmune or vasculitis"
       for code in [3016,3017,3018,3019,3033,3053,3055,3056,3057]},
    3059: "Kidney Nephrolithiasis",
    3060: "Kidney Urolithiasis",
    3073: "Kidney Lithium toxicity",
    3074: "Kidney HIV nephropathy",
    **{code: "Kidney Other" for code in [3051,3054,3062,3067,3068,3072]},
    **{code: "Other" for code in
        list(range(0, 2000)) +
        list(range(4000, 6500))              
    },
    -9999: "MISSING",
}

# format: MALIG
MALIG_MAP = {
    512:           "Unknown malignancy",
    **{code: "Skin cancer" for code in [1, 2]},             
    4:              "CNS tumor",
    8:              "Genitourinary cancer",
    16:             "Breast cancer",
    32:             "Thyroid cancer",
    64:             "Head & neck cancer",
    128:            "Lung cancer",
    256:            "Hematologic malignancy",             
    **{code: "Liver cancer" for code in [2048, 4096, 8192]},  
    1024:           "Other, specify",
    -9999: "MISSING",
}

# Format: HSTSTST  
HSTSTST_MAP = {
    # 1No prior cancer
    1: "No prior cancer",
    # kin cancers
    **{code: "Skin cancer" for code in [2, 3]},
    # CNS tumors
    **{code: "CNS tumor" for code in [4, 5, 6, 7, 8, 9, 12]},
    # Genitourinary cancers
    **{code: "Genitourinary cancer" for code in list(range(13, 23))},  
    # Gastrointestinal cancers
    **{code: "Gastrointestinal cancer" for code in list(range(23, 29))},  
    # Breast & thyroid cancers
    **{code: "Breast/Thyroid cancer" for code in [29, 30]},
    # Head & neck / lung cancers
    **{code: "Head & neck / Lung cancer" for code in [32, 33, 34]},
    # Leukemia / Lymphoma
    35: "Leukemia/Lymphoma",
    998: "Unknown",
    999: "Other, specify",
    -9999: "MISSING",
}

# Format: THCOD 
THCOD_MAP = {
    # Graft failure
    **{c: "Graft failure" for c in [2000, 2001, 2002, 2003, 2004, 2005, 2006, 2099]},
    # Infection
    **{c: "Infection" for c in [
        2100, 2101, 2109, 2110, 2111, 2112, 2119,
        2120, 2129, 2130, 2198, 2199
    ]},
    # Cardiovascular
    **{c: "Cardiovascular" for c in list(range(2200, 2210)) + [2299]},
    # Pulmonary
    **{c: "Pulmonary" for c in list(range(2300, 2306)) + [2399]},
    # Cerebrovascular
    **{c: "Cerebrovascular" for c in list(range(2400, 2404)) + [2499]},
    # Hemorrhage
    **{c: "Hemorrhage" for c in list(range(2500, 2505)) + [2599]},
    # Malignancy
    **{c: "Malignancy" for c in [2600, 2601, 2602, 2603, 2604, 2699]},
    # Other medical (diabetes, renal/liver failure, drug toxicity, etc.)
    **{c: "Other medical" for c in list(range(2700, 2714))},
    # Trauma / Other external
    **{c: "Trauma/other external" for c in [2800, 2801, 2802, 2803]},
    998: "Unknown",
    999: "Other, specify",
    -9999: "MISSING",
}

# Format: FOLCD  
FOLCD_MAP = {
    1: "Hospital discharge",
    # <1 year
    **{c: "Short-term (<1 yr)" for c in (3, 6)},
    # 1–5 years
    **{c: "Early follow-up (1–5 yr)" for c in (10, 20, 30, 40, 50)},
    # 6–10 years
    **{c: "Mid follow-up (6–10 yr)" for c in (60, 70, 80, 90, 100)},
    # 11–20 years
    **{c: "Late follow-up (11–20 yr)" for c in range(110, 210, 10)},
    # >20 years
    **{c: "Long-term (>20 yr)" for c in range(210, 800, 10)},
    # All graft-failure time-points (800–850)
    **{c: "Graft-failure follow-up" for c in range(800, 851)},
    # Lost / died  
    900: "Lost to follow-up",  
    998: "Lost to follow-up",  
    999: "Recipient death",
    -9999: "MISSING",
}

# KICOD  
KICOD_MAP = {
    # Unknown / Other
    998: "Unknown",
    999: "Other",
    # Graft failure
    **{code: "Graft failure" for code in [3200, 3201, 3202, 3203, 3204, 3299]},
    # Infection
    **{code: "Infection" for code in [
        3300, 3301, 3302, 3303, 3304, 3305, 3306, 3307, 3308, 3399
    ]},
    # Cardiovascular
    **{code: "Cardiovascular" for code in [3400, 3401, 3402, 3499]},
    # Cerebrovascular
    **{code: "Cerebrovascular" for code in [3500, 3599]},
    # Hemorrhage
    **{code: "Hemorrhage" for code in [3600, 3601, 3699]},
    # Malignancy
    **{code: "Malignancy" for code in [3700, 3701, 3702, 3799]},
    # Trauma
    **{code: "Trauma" for code in [3800, 3899]},
    # Miscellaneous
    **{code: "Miscellaneous" for code in [
        3900, 3901, 3902, 3903, 3904, 3905,
        3906, 3907, 3908, 3909, 3910, 3911,
        3912, 3913, 3914, 3915
    ]},
    -9999: "MISSING",
}

# Mappings for group c_flag_other

# for DON_GENDER, CAN_GENDER
GENDER_MAP = {
    "M": "Male",
    "F": "Female",
    "-9999": "MISSING",
}

# for CAN_SOURCE
CAN_SOURCE_MAP = {
    "A": "Active Wailtlist",
    "R": "Removed Waitlist",
    "L": "Live Don Organ Recipient never on waitlist",
    "-9999": "MISSING",
}


# Mappings for group e_char_with_encoding

# for CAN_ABO, DON_ABO
ABO_MAP = {
    "A": "A",
    "A1": "A1",
    "B": "B",
    "A1B": "A1B",
    "A2": "A2",
    "A2B": "A2B",
    "AB": "AB",
    "O": "O",
    "Z": "Z",
    "UNK": "Unkown",
    -9999: "MISSING",
}

# for CAN_RACE_SRTR, DON_RACE_SRTR
RACEBSR_MAP = {
    "ASIAN": "Asian",
    "BLACK": "Black",
    "MULTI": "Multiracial",
    "NATIVE": "Native American",
    "PACIFIC": "Pacific Islander",
    "WHITE	": "White",
    "-9999": "MISSING",
}

# for DON_ANTI_CMV, DON_ANTI_HCV, DON_CMV_IGG, TFL_CMV_IGG, TFL_CMV_IGM	
SRLSTT_MAP = {
    "C": "Cannot Disclose",
    "I": "Indeterminate",
    "N": "Negative",
    "ND": "Not Done",
    "P": "Positive",
    "PD": "Pending",
    "U": "Unknown",
    "-9999": "MISSING",
}

# for REC_PX_STAT
PXSTATA_MAP = {
    "A": "Living",
    "D": "Dead",
    "L": "Lost",
    "R": "Retransplanted",
    "U": "Unknown",
    "X": "Natural Disaster",
    "-9999": "MISSING",
}

# for TFL_PX_STAT, TFL_LASTATUS
PXSTATB_MAP = {
    "A": "Living",
    "D": "Dead",
    "L": "Lost to FU",
    "N": "Not Seen",
    "R": "Retransplanted",
    "-9999": "MISSING",
}

# Mappings for group d_char_free_text

# for REC_CMV_IGG, REC_CMV_IGM, REC_CMV_STAT, REC_EBV_STAT, REC_HBV_ANTIBODY, REC_HBV_SURF_ANTIGEN, REC_HCV_STAT
# no format available in SRTR Data Dictionary ->Sorted for unique values then matched values based on previous analysis
STAT_D_MAP = {
    "P": "Positive",
    "N": "Negative",
    "ND": "Not Done",
    "U": "Unknown",
    "-9999": "MISSING",
}