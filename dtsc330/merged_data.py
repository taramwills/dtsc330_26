import pandas as pd
from dtsc330.readers import articles, grants

def get_merged_data(grant_path: str = 'data\RePORTER_PRJ_C_FY2025.zip',
                    article_path: str = 'data\pubmed25n1275.xml.gz') -> pd.DataFrame:
    art = articles.Articles(article_path)
    auth_df = art.get_authors()

    grant = grants.Grants(grant_path)
    grant_df = grant.get_grantees()

if __name__ == '__main__':
    get_merged_data()