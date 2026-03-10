# dtsc330

# HW 2

How do you fill in the missing dates from the grants data?
    - Made loop to look at range of missing dates
        ○ Rows 65925-70459
    - Trouble finding pattern
        ○ Out of order
    - Decided to use the average date for NaNs
        ○ Looked for last dates entered between NaNs and filled in midpoint date

PI_NAMEs contains multiple names. We can only connect individual people. Can you make it so that we can get "grantees"?
Split PI_NAMES into different rows using pandas
    - Clean pi names
        ○ Remove (contact)
        ○ Get rid of whitespace
        ○ Determine delimiters
    - Split pi names into list of individual names
        ○ Used explode() to create one row per investigator
    - Removed empty/missing entries and retained application ID between each person and their grant
    - Create normalized grantees table with one person per row

The dates for Articles are problematic. Can you fix them?
    - Look at how dates are formatted
    - Determine what tag we're parsing
        ○ PubDate vs. others
    - If PubDate tag: extra loop
        ○ If tags are month, day, year: save into variable
Format into year-month-day

# HW 3
- Read in har (first 10 participants) and reusable classifier files
- Converted the timestamp data into a timeseries so that I could resample the data
- Determined unique participants
- For each person, computed acceleration magnitude using acc_x, acc_y, acc_y
- Resampled into 10 or 60 second intervals
- Determined features (acc_mag) and labels (is_sleep)
- Used first participants as test set and remaining in training set
- After loop, read in classifier using random forest model type
- Train features and labels
- Predict test features
- Print out accuracy
- 60s: 0.5862884160756501
- 10s: 0.6007462686567164

# HW 4
After implementing XGBoost to my reusable classifier, these are the new accuracies:
10s: 
Correct: 1484 / 2412
Accuracy: 0.615257048092869

60s:
Correct: 242 / 423
Accuracy: 0.5721040189125296
The 10s interval became slightly more accurate and the 60s interval became slightly less accurate.

![Machine learning diagram](actual-machine-learning-diagram.png)

# HW 5

A way that you could match the phonebooks is by comparing area codes to people of the same names. If there are two people with the same area code, then see if the addresses are the same.