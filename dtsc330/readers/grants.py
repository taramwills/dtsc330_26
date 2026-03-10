import pandas as pd
import sqlalchemy

class Grants(): # Class names in python 
    def __init__(self, path: str | None = None):
        """Create and parse a Grants fuke
        
        Args:
            path (str): the location of the file on the disk
            if empty, defaults to pulling from database
        """
        # What is self?
        # "Self is the specific instance of the object"
        # Store shared variables in self
        if path is None:
            pass
        self.path = path
        self.df = self._parse(path)

    def _parse(self, path: str):
        """Parse a grants file"""
        df = pd.read_csv(path, compression = 'zip')

        # Q2
        # Create copy of df
        g = df[['APPLICATION_ID', 'PI_NAMEs']].copy()

        # Remove '(contact)'
        g['PI_NAMEs'] = g['PI_NAMEs'].str.replace(r'\s*\(contact\)\s*', '', regex = True)

        # Make sure separators look the same
        g['PI_NAMEs'] = g['PI_NAMEs'].str.replace(r'\s*;\s*', ';', regex = True)

        # Strip whitespace
        g['PI_NAMEs'] = g['PI_NAMEs'].str.strip()

        # test = g['PI_NAMEs'].head()
        # print(test)

        g['grantee'] = g['PI_NAMEs'].str.split(';')
        g = g.explode('grantee')

        test2 = g['grantee']
        # print(test2)

        # Strip whitespace
        g['grantee'] = g['grantee'].str.strip()

        # Get rid of empty strings
        g = g.dropna(subset = ['grantee'])
        g = g[g['grantee'] != '']

        print(test2)

        # Q1
        # Create variables
        col = 'BUDGET_START'
        start_row = 65925
        end_row = 70459

        # Make sure all dates in same format
        df[col] = pd.to_datetime(df[col], errors = 'coerce')

        # Work on just the slice (positional rows)
        s = df[col].iloc[start_row:end_row + 1].copy()

        # Identify NaT runs (consecutive missing segments)
        is_missing = s.isna()
        run_id = (is_missing != is_missing.shift()).cumsum()

        for rid, idxs in s[is_missing].groupby(run_id).groups.items():
            # idxs is an Index of slice-relative indices (same as s.index values)
            first_idx = idxs.min()
            last_idx = idxs.max()

            # last known date BEFORE the missing run (within the slice)
            prev_known = s.loc[:first_idx].dropna()
            if prev_known.empty:
                # If there's no known date before within the slice, look outside (before start_row)
                prev_known = df[col].iloc[:start_row].dropna()
                if prev_known.empty:
                    raise ValueError(f"No previous known date found before missing run starting at index {first_idx}")
                prev_date = prev_known.iloc[-1]
            else:
                # Note: prev_known includes values up to first_idx; since first_idx is missing, this is safe
                prev_date = prev_known.iloc[-1]

            # next known date AFTER the missing run (within the slice)
            next_known = s.loc[last_idx:].dropna()
            if next_known.empty:
                # If there's no known date after within the slice, look outside (after end_row)
                next_known = df[col].iloc[end_row + 1:].dropna()
                if next_known.empty:
                    raise ValueError(f"No next known date found after missing run ending at index {last_idx}")
                next_date = next_known.iloc[0]
            else:
                # next_known includes last_idx; since last_idx is missing, first non-null will be after it
                next_date = next_known.iloc[0]

            # midpoint
            midpoint = (prev_date + (next_date - prev_date) / 2).normalize()  # normalize -> midnight date

            # fill entire run with same midpoint
            s.loc[idxs] = midpoint

        # Write back filled slice into df
        df.loc[s.index, col] = s

        missing_dates = df["BUDGET_START"].iloc[65925:70459 + 1].isna().sum()
        # print(f"Missing budget_start in range: {missing_dates}")

        new_dates = df[["BUDGET_START"]].iloc[65920:65940]
        print(new_dates)

        # Initialize amount of missing dates
        missing_count = 0

        # Finds rows where missing dates are in BUDGET_START
        for i, value in enumerate(df['BUDGET_START']):
            if pd.isna(value):
                missing_count += 1

                # Print row number and a few identifying fields
                app_id = df.loc[i, 'APPLICATION_ID'] if 'APPLICATION_ID' in df.columns else None
                pi = df.loc[i, 'PI_NAMEs'] if 'PI_NAMEs' in df.columns else None
                org = df.loc[i, 'ORG_NAME'] if 'ORG_NAME' in df.columns else None

                print(f"Row {i}: missing | APPLICATION_ID = {app_id} | PI_NAMEs = {pi} | ORG_NAME = {org}")

        # print(f"\nTotal missing: {missing_count}")

        mapper = {
            'APPLICATION_ID': 'application_id',
            'BUDGET_START': 'start_at', # _at means a date
            'ACTIVITY': 'grant_type',
            'TOTAL_COST': 'total_cost',
            'PI_NAMEs': 'pi_names',
            'ORG_NAME': 'organization',
            'ORG_CITY': 'city',
            'ORG_STATE': 'state',
            'ORG_COUNTRY': 'country'
            }
        
        # Make column names lowercase
        # Maybe combine for budget duration
        df = df.rename(columns = mapper)[mapper.values()]

        return df
    
    def to_db(self, path: str = 'data/article_grant_db.sqlite'):
        """Send read-in data to the database

        Args:
            path (str, optional): Location of sqlite file
            defaults to 'data/article_grant_db.sqlite'
        
        """
        # Define connection
        engine = sqlalchemy.create_engine('sqlite:///data/article_grant_db.sqlite')
        connection = engine.connect()

        # Always append, deletion should be more thoughful
        # NEVER ALTER RAW DATA
        # You can use primary key as index -> complicated
        self.df[['application_id',
                 'start_at',
                 'grant_type',
                 'total_cost']].to_sql('grants', connection, if_exists = 'append', index = False)

    def _from_db(self):
        """Load data from database"""
        engine = sqlalchemy.create_engine('sqlite:///data/article_grant_db.sqlite')
        connection = engine.connect()
        df = pd.read_sql('SELECT * FROM grants', connection)
        return df

    def get_grants(self):
        """Get parsed grants"""
        return self.df
    
    def get_grantees(self):
        """Get parsed grantees"""
        pass

if __name__ == '__main__':
    # This is for debugging
    grants = Grants('data/RePORTER_PRJ_C_FY2025.zip')
    grants.to_db()