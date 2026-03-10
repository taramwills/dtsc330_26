import pandas as pd
import numpy as np
# import fasttext
from dtsc330.readers import articles, grants

class Phonebook():
    """
    Authors DF - first name, last name, initials, affiliation
    Grantees DF - same columns
    """
    def __init__(self, path: str):
        self.art_df = self.retrieve_articles()
        self.grant_df = self.retrieve_grants()
        self.prematched_df = self.prematch()

    def retrieve_articles(self):
        article = articles.Articles()
        return article.get_authors()
    
    def retrieve_grants(self):
        grant = grants.Grants()
        return grant.get_grants()
    
    def prematch(self):
        return self.art_df.merge(self.grant_df,
                                 on = ['first_name', 'last_name', 'affiliation'],
                                 how = 'inner')
        
    def combine_df(self):
        combined_df = pd.DataFrame()

        for i, row1 in self.art_df.iterrows():
            for j, row2 in self.grant_df.iterrows():
                new_row = pd.concat([row1, row2])
                combined_df.add(new_row)

        return combined_df


if __name__ == '__main__':
    pb = Phonebook()