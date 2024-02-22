# DOC
Disentangled Opinion Clustering

Please follow the instructions to download the datasets and reproduce the results:

## Download Configurations and Datasets:
1. Download [input.zip](https://doc-inputzip.s3.eu-west-1.amazonaws.com/input.zip), unzip it and place the contents into ./DOCClusterRepLearning/input
2. Download [moral-foundation-dataset.zip](https://doc-inputzip.s3.eu-west-1.amazonaws.com/moral-foundation-dataset.zip), unzip it and place the contents into ./moral-foundation-dataset
3. Download [ckpt_deberta-v3-large-auto_0.bin](https://doc-inputzip.s3.eu-west-1.amazonaws.com/CMF-save/ckpt_deberta-v3-large-auto_0.bin) and place it into ./DOCClusterRepLearning/input/5-fold-220727/deberta-v3-large-auto. Download [ckpt_deberta-v3-large-auto_0.bin](https://doc-inputzip.s3.eu-west-1.amazonaws.com/VAD-save/ckpt_deberta-v3-large-auto_0.bin) and place it into ./DOCClusterRepLearning/input/5-fold-220730. Please be aware that they are different files, the first is for CMF and the second is for VAD, so please follow the link order and don't mix them up.

## Pre-process the datasets:

4. Look at ./DOCClusterRepLearning/datasets/CMF_train_r_withouttext.csv. The text column is anonymised. So please restore the text field using the twitterID before pythoning. **Reminder**: (1) Since both CMF (https://gitlab.com/mlpacheco/covid-moral-foundations/-/tree/main) and VAD are twitter datasets, some tweets might have been deleted at the time of your reproduction. So be prepared for missing some of the tweets. (2) Make sure you obtain full_text when collecting tweets using the TwitterAPI.

5. After you've restored the text, rename the files to `CMF_test_r.csv`, `CMF_train_r.csv`, `VAD_test.csv`, `VAD_train.csv`.

6. `cd ./DOCClusterRepLearning/src` and run `python pairwise-traintestinstance-construct-CMF.py` to create pairwise training instances.

## Clustering:

7. When you first run the model, you will encounter an error that "MODEL_PATHS[self.name] is None, use online config value...". Don't panic, just comment out line 81, and uncomment line 82. Then everything will be fine.

8. For CMF dataset, `cd ./DOCClusterRepLearning/src` and type `python doctrain_deberta_auto_CMF.py` to run the model. Uncomment line 3361 and line 3375, comment out line 3378 to run the baseline model.

9. For VAD dataset, `cd ./DOCClusterRepLearning/src` and type `python doctrain_deberta_auto_VAD.py` to run the model. Uncomment line 3104-3118, comment out line 3131 to run the baseline model.

## Training:
10. For both datasets, comment out `clustering()` and uncomment `main()` in the `if __name__ == '__main__':` entry.
