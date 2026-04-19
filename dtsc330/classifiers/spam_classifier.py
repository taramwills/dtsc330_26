from reusable_classifier import ReusableClassifier as rc
import pandas as pd

class SpamClassifier(rc):
    def __init__(self, model_type: str = 'xgboost'):
        self.classifier = rc(model_type = model_type)
        self.data = None
        self.features = None
        self.labels = None

    def load_data(self, path: str):
        """Load in the data from a path"""
        self.data = pd.read_csv(path)

    def prepare_data(self):
        """Prepare the data for training and testing."""
        df = self.data.copy()

        # 'subject' column has these values that are indicative of spam emails
        subject_spam_words = ['WIN', 'selected', 'URGENT', 'limited']

        # Create features for each word
        for word in subject_spam_words:
            df[f'has_{word}_in_subject'] = df['subject'].str.contains(word).astype(int)

        # 'email_text' column has these values that are indicative of spam emails
        email_spam_words = ['free', 'offer', 'click', 'buy', 'cash', 'guarantee']

        # Create features for each word
        for word in email_spam_words:
            df[f'has_{word}_in_email'] = df['email_text'].str.contains(word).astype(int)

        # Set features
        self.features = df[['has_WIN_in_subject',
                            'has_selected_in_subject',
                            'has_URGENT_in_subject',
                            'has_limited_in_subject',
                            'has_free_in_email',
                            'has_offer_in_email',
                            'has_click_in_email',
                            'has_buy_in_email',
                            'has_cash_in_email',
                            'has_guarantee_in_email']]
        
        self.labels = df['label']

if __name__ == '__main__':
    sc = SpamClassifier(model_type = 'xgboost')
    sc.load_data('data/spam_email_dataset.csv')
    sc.prepare_data()

    print(sc.features.head())
    print(sc.labels.head())

    score = sc.classifier.assess(sc.features, sc.labels)
    print('Score:', score)
    print('Done')