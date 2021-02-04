import re
import pandas as pd

class SimilarityJoin:
    def __init__(self, data_file1, data_file2):
        self.df1 = pd.read_csv(data_file1)
        self.df2 = pd.read_csv(data_file2)

    def preprocess_df(self, df, cols): 
        # adds the "joinKey" column to the input
        df["joinKey"] = df[cols[0]].fillna('') + " " + df[cols[1]].fillna('')
        df['joinKey'] = df['joinKey'].str.lower()
        df["joinKey"] = df['joinKey'].str.split(r'\W+')

        return df

    def filtering(self, df1, df2):
        original_df1 = df1
        original_df2 = df2
        # flatten the record
        df1 = df1.explode('joinKey')
        df1 = df1[["id", "joinKey"]]
        # replace empty string with NA
        df1['joinKey'].replace('', np.nan, inplace=True)
        # delete the empty string
        df1.dropna(subset=['joinKey'], inplace=True)

        df2 = df2.explode('joinKey')
        df2 = df2[["id", "joinKey"]]
        df2['joinKey'].replace('', np.nan, inplace=True)
        df2.dropna(subset=['joinKey'], inplace=True)
        # join two df
        merge_result = df1.merge(df2, on="joinKey", suffixes=('_1', '_2'))
        # remove duplicates
        merge_result = merge_result.drop_duplicates(subset=['id_1', 'id_2'], keep='first')
        # join with the original df to get the joinKey
        res_1 = merge_result.merge(original_df1, left_on="id_1", right_on="id", suffixes=(None, '_1'))
        res_1 = res_1[["id_1", "joinKey_1", "id_2"]]

        res_2 = res_1.merge(original_df2, left_on="id_2", right_on="id")
        res_2 = res_2[["id_1", "joinKey_1", "id_2", "joinKey"]]
        res_2 = res_2.rename(columns={"joinKey": "joinKey_2"})
        return res_2

    def verification(self, cand_df, threshold):
        joinKey_1 = cand_df['joinKey_1']
        joinKey_2 = cand_df['joinKey_2']
        cand_df['jaccard'] = self.helper(joinKey_1, joinKey_2)
        final_df = cand_df.loc[(cand_df['jaccard'] >= threshold)]
        return final_df
    # helper function to calculate jaccard
    def helper(self, joinKey_1, joinKey_2):
        keys_1 = joinKey_1.values.tolist()
        keys_2 = joinKey_2.values.tolist()

        n = len(keys_1)
        res_list = []
        for i in range(n):
            # remove empty string
            new_keys_1 = list(filter(None, keys_1[i]))
            new_keys_2 = list(filter(None, keys_2[i]))
            new_keys_list = list(set(new_keys_1 + new_keys_2))
            # dictionary to store all keys in the first joinKey_1
            dict = {}
            count = 0
            # use set to avoid count the same key twice
            visited = set()
            for key in new_keys_1:
                dict[key] = 1
            for key in new_keys_2:
                if key in dict and key not in visited:
                    count += 1
                    visited.add(key)
            jaccard = count / len(new_keys_list)
            res_list.append(jaccard)
        sentence_series = pd.Series(res_list)
        return sentence_series

    def evaluate(self, result, ground_truth):
        true_match = 0
        for record in result:
            if record in ground_truth:
                true_match += 1
        precision = true_match / len(result)
        recall = true_match / len(ground_truth)
        fmeasure = (2 * precision * recall) / (precision + recall)
        return (precision, recall, fmeasure)

    def jaccard_join(self, cols1, cols2, threshold):
        new_df1 = self.preprocess_df(self.df1, cols1)
        new_df2 = self.preprocess_df(self.df2, cols2)
        print ("Before filtering: %d pairs in total" %(self.df1.shape[0] *self.df2.shape[0])) 

        cand_df = self.filtering(new_df1, new_df2)
        print ("After Filtering: %d pairs left" %(cand_df.shape[0]))

        result_df = self.verification(cand_df, threshold)
        print ("After Verification: %d similar pairs" %(result_df.shape[0]))

        return result_df



if __name__ == "__main__":
    er = SimilarityJoin("Amazon_sample.csv", "Google_sample.csv")
    amazon_cols = ["title", "manufacturer"]
    google_cols = ["name", "manufacturer"]
    result_df = er.jaccard_join(amazon_cols, google_cols, 0.5)

    result = result_df[['id_1', 'id_2']].values.tolist()
    ground_truth = pd.read_csv("Amazon_Google_perfectMapping_sample.csv").values.tolist()
    print ("(precision, recall, fmeasure) = ", er.evaluate(result, ground_truth))