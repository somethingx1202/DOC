# DOC
Disentangled Opinion Clustering

Please follow the steps to download the datasets and reproduce the results:

## Download Configurations and Datasets:
1. Download [input.zip](https://doc-inputzip.s3.eu-west-1.amazonaws.com/input.zip), unzip it and place the contents into ./DOCClusterRepLearning/input
2. Download [moral-foundation-dataset.zip](https://doc-inputzip.s3.eu-west-1.amazonaws.com/moral-foundation-dataset.zip), unzip it and place the contents into ./moral-foundation-dataset
3. Download [ckpt_deberta-v3-large-auto_0.bin](https://doc-inputzip.s3.eu-west-1.amazonaws.com/CMF-save/ckpt_deberta-v3-large-auto_0.bin) and place it into ./DOCClusterRepLearning/input/5-fold-220727/deberta-v3-large-auto. Download [ckpt_deberta-v3-large-auto_0.bin](https://doc-inputzip.s3.eu-west-1.amazonaws.com/VAD-save/ckpt_deberta-v3-large-auto_0.bin) and place it into ./DOCClusterRepLearning/input/5-fold-220730. Please be aware that they are different files, the first is for CMF and the second is for VAD, so please follow the link order and don't mix them up.

## Pre-process the datasets:

4. Look at ./DOCClusterRepLearning/datasets/CMF_train_r_withouttext.csv. The text column is anonymised. So please restore the text field using the twitterID before pythoning. **Reminder**: (1) Since both CMF and VAD are twitter datasets, some tweets might have been deleted at the time of your reproduction. So be prepared for missing some of the tweets. (2) Make sure you obtain full_text when collecting tweets using the TwitterAPI.

5. After you've restored the text, rename the files to `CMF_test_r.csv`, `CMF_train_r.csv`, `VAD_test.csv`, `VAD_train.csv`.

6. `cd ./DOCClusterRepLearning/src` and run `python pairwise-traintestinstance-construct-CMF.py` to create pairwise training instances.

## Clustering:

7. When you first run the model, you will encounter an error that "MODEL_PATHS[self.name] is None, use online config value...". Don't panic, just comment out line 81, and uncomment line 82. Then everything will be fine.

8. For CMF dataset, `cd ./DOCClusterRepLearning/src` and type `python doctrain_deberta_auto_CMF.py` to run the model. Uncomment line 3361 and line 3375, comment out line 3378 to run the baseline model.

9. For VAD dataset, `cd ./DOCClusterRepLearning/src` and type `python doctrain_deberta_auto_VAD.py` to run the model. Uncomment line 3104-3118, comment out line 3131 to run the baseline model.

## Training:
10. For both models, comment out `clustering()` and uncomment `main()` in the `if __name__ == '__main__':` entry.




The annotations are given in the form of `ID,stance,aspect_span_start:aspect_span_end,opinion_span_start:opinion_span_end,aspect_catetegory` <br />
in the `Datasets_Raw` folder. <br />

To obtain tweet text,

1.  `cd twitter_get_text_by_id_twitter4j`
2.  Open `./settings/crawler.properties` and setup your `consumerKey, consumerSecret, access token and access token secret`.
    1. For the acquisition of `consumerKey, consumerSecret, access token and access token secret`, please refer to https://developer.twitter.com/en/docs/developer-portal/overview. The Standard v1.1 is sufficient.
3.  run twitter_get_text_by_id_twitter4j by either `java -jar twitter_vac_opi_cwl_by_id.jar ./settings/crawler.properties` or `javac -cp "./*" ./src/main/org/backingdata/twitter/crawler/rest/TwitterRESTTweetIDlistCrawler.java` The tweets are stored in `./saves` in json format.

## VAD unsupervised training
`cd VADMlmFineTuning`<br />
VADtransformer is firstly trained unsupervised. The model will be saved to `../datasets/mlm-vad`. 

To perform unsupervised training,

1.  Replace tweetIDs in `UnannotatedTwitterID_training.csv` and `UnannotatedTwitterID_testing.csv` with obtained tweet text.
2.  Put the tweet text file in `../datasets`. The format is the same as `vad_train_finetune.txt`.
3.  `cd src` and run `train_vad_albert_vae.py`

## VAD supervised training
`cd VADStanceAndTextspanPrediction`

In the previous step we obtain the unsupervised pre-trained VAD, scilicet the TopicDrivenMaskedLM. At this stage we wrap the model with classifiers and constrains, and train the model.

To perform supervised training,

1.  Move the saved model (i.e., the `pytorch_model.bin` file) from the `../datasets/mlm-vad` of **VAD unsupervised training** to the `./datasets/albertconfigs/vadlm-albert-large-v2/vad-cache` folder. For your convenience a saved TopicDrivenMaskedLM is ready-to-use in the `./datasets/albertconfigs/vadlm-albert-large-v2/vad-cache` folder.
2.  Move the saved config of the model (i.e., the `config.json` file) from the `../datasets/mlm-vad` of **VAD unsupervised training** to the `./datasets/albertconfigs/vadlm-albert-large-v2/vadlm-albert-large-v2` folder. For your convenience a saved config.json is ready-to-use in the `./datasets/albertconfigs/vadlm-albert-large-v2/vadlm-albert-large-v2` folder.
3.  `cd src` and run `vadtrain_eval_predict.py` for training and testing.
    1.  Training: Uncomment line 1559-1578 of `vadtrain_eval_predict.py` and run the file. Checkpoints will be saved in `./datasets/vadcheckpoints/5-fold-211103/vadlm-albert-large-v2/`
    2.  Testing: Uncomment line 1580-1608 of `vadtrain_eval_predict.py` and run the file. The prediction will be output in same directory. A saved model can be downloaded via this [link](https://vadsupsave.s3.eu-west-2.amazonaws.com/vadsupervisedsavedmodels.zip). You can place the save model in `./datasets/vadcheckpoints/5-fold-211103/vadlm-albert-large-v2/` for a quick start.

