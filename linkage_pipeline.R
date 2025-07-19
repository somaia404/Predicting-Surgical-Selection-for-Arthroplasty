# linkage_pipeline.R
# Author: Somaia Elsheikhi
# Purpose: Link SMR01, OPERA, PROMs, and Radiology report data for hip replacement patients

# -------------------- Load libraries --------------------
library(dplyr)
library(readr)
library(lubridate)

# -------------------- Step 1: Load datasets --------------------
smr01 <- read_csv("smr01_linked.csv")
opera <- read_csv("opera_hip_operations.csv")
proms <- read_csv("proms_hip_data_200patients.csv")
radiology <- read_csv("hip_radiology_reports.csv")

# -------------------- Step 2: Clean and standardise --------------------
# Rename CHI column
proms <- proms %>% rename(CHI = CHI_NO)

# Standardise CHI and remove white space
smr01$CHI <- trimws(as.character(smr01$CHI))
opera$CHI <- trimws(as.character(opera$CHI))
proms$CHI <- trimws(as.character(proms$CHI))
radiology$CHI <- trimws(as.character(radiology$CHI))

# Remove duplicates
proms <- proms %>% distinct(CHI, .keep_all = TRUE)
radiology <- radiology %>% distinct(CHI, .keep_all = TRUE)

# Remove rows with missing key values
proms <- proms %>% filter(!is.na(CHI) & !is.na(eq_vas))
radiology <- radiology %>% filter(!is.na(CHI) & !is.na(MAIN_OPERATION_DATE) & !is.na(ReportText))

# Standardise surgical side field
opera <- opera %>%
  mutate(SIDE = toupper(trimws(SIDE)))

# Safely convert date columns
smr01$MAIN_OPERATION_DATE <- suppressWarnings(ymd(smr01$MAIN_OPERATION_DATE))
radiology$Report_Date <- suppressWarnings(ymd(radiology$MAIN_OPERATION_DATE))

# -------------------- Step 3: Identify surgical cohort --------------------
surgical_cohort <- opera %>%
  select(CHI, MAIN_OPERATION, MAIN_OPERATION_DATE, SIDE)

# -------------------- Step 4: Merge surgical cohort with SMR01 --------------------
surgery_smr01 <- surgical_cohort %>%
  left_join(smr01, by = "CHI", suffix = c("_opera", "_smr01")) %>%
  mutate(Surgery_Date = MAIN_OPERATION_DATE_opera)

# -------------------- Step 5: Link to PROMs data --------------------
proms_outcome <- proms %>%
  filter(Scoring_Point == "6months" & !is.na(eq_vas))

surgery_proms <- surgery_smr01 %>%
  left_join(proms_outcome, by = "CHI")

# -------------------- Step 6: Link to radiology reports (pre-op only) --------------------
final_linked <- surgery_proms %>%
  left_join(radiology, by = "CHI") %>%
  filter(Report_Date < Surgery_Date)

# -------------------- Step 7: Export final linked dataset --------------------
final_dataset <- final_linked %>%
  select(CHI, Surgery_Date, SIDE, MAIN_OPERATION, eq_vas, Report_Text)

write_csv(final_dataset, "linked_clean_dataset.csv")
