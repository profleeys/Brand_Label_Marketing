from cgitb import text
from cmath import nan
import streamlit as st
import re
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
import plotly.express as px
import seaborn as sns
import altair as alt
import os
st.set_page_config(layout="wide")
###
path = os.getcwd()
feature_df = pd.read_csv(path+'/Desktop/streamlit_app/Feature_small.csv')
mark_col = pd.read_csv(path+'/Desktop/streamlit_app/Mark_small.csv')
model = joblib.load(path+'/Desktop/streamlit_app/dt_1step_model.pkl')
model_1 = joblib.load(path+'/Desktop/streamlit_app/dt_2step_model.pkl')
model_2 = joblib.load(path+'/Desktop/streamlit_app/dt_1step_model.pkl')
###
mark_col_name = mark_col.columns
feature_col_name = feature_df.columns
text = st.sidebar.text_input('輸入句子')
###
row0_spacer1, row0_1, row0_spacer2, row0_2, row0_spacer3 = st.columns((.5, 5, .1, 1.3, .1))
with row0_1:
    st.title('判斷句子會算是哪個標籤')
with row0_2:
    st.text("")
row3_spacer1, row3_1, row3_spacer2 = st.columns((.1, 3.2, .1))
with row3_1:
    st.text('')
    
st.header('標籤預測')
row1_spacer1, row1_1, row1_spacer2 = st.columns((.2, 7.1, .2))
with row1_1:
    st.subheader('Choose the Algorithm:')
    
row2_spacer1, row2_1, row2_spacer2, row2_2, row2_spacer3  = st.columns((.2, 2.3, .4, 4.4, .2))
with row2_1:
    st.subheader('Here has two Algorithm')    
    alg = st.radio ("Algorithm Choice", ['Decision Tree'])
    
with row2_2:
    if alg == 'Decision Tree':
        st.subheader('Here has two step you can choose!')
        step = st.radio ("Which step you want to use?", ['1','2','3'])
    else:
        " "
if alg == 'Decision Tree':
    if step == '1':
        if st.button("start Predict ",key=1):   
            #load txt
            rule = []
            string = '(twitter)'
            f = open(path+'/Desktop/streamlit_app/sort 分析詞_1105.txt', encoding='UTF-8') 
            lines = f.readlines()
            for line in lines:
                line = line.strip('\n')
                string = string +'|'+ str(line)
            f.close()
            #feature
            feature = []
            a = re.findall(string,str(text), flags=re.IGNORECASE)
            s = ''
            for j in a :
                s = ''.join(j)+' '+s
            feature.append(s)

            # tfidf
            vectorizer2 = TfidfVectorizer(smooth_idf=True)
            #得到TF Matrix
            feature_empty = ['']
            if feature == feature_empty:
                fin_tfVec = pd.DataFrame(columns=feature_col_name)
                if fin_tfVec.empty:
                    fin_tfVec.loc[0] = 0
            else:
                tfidf = vectorizer2.fit_transform(feature)
            #词的对应索引值
                tf_feature_name = vectorizer2.get_feature_names_out() 
                tfVec = pd.DataFrame(tfidf.toarray(),columns=tf_feature_name)
                fin_tfVec = pd.DataFrame(data=tfVec,columns=feature_col_name)
                fin_tfVec.replace(np.nan,0,inplace=True)


            pred = model.predict(fin_tfVec)
            pred_df = pd.DataFrame(pred,columns=mark_col_name)
            pred_prob = model.predict_proba(fin_tfVec)
            ###
            train_not_mark = pd.DataFrame(columns=mark_col_name)
            for i in range(pred.shape[0]):
                if pred_df.loc[i].sum() == 0 :
                    tmp = pd.DataFrame(pred_df.iloc[i])
                    train_not_mark = pd.concat([train_not_mark,tmp.T])
            pred_copy = pred.copy()
            for i in range(train_not_mark.shape[0]):
                a= pd.DataFrame(pred_df.loc[train_not_mark.index[i]])
                a=a.T
                prediction_of_probability = model.predict_proba(a)
                x = np.array(prediction_of_probability)[:,:,1]                
                for j in range(0,len(x)) :
                    if np.max(x) == x[j]:
                        pred_copy.loc[a.index[0]][j]=1
            ###
            st.subheader("查看各個標籤的機率值")
    
            pred_prob_df = pd.DataFrame()
            for i in range(len(pred_prob)):
                tmp = pd.DataFrame(pred_prob[i]).T
                pred_prob_df=pd.concat([pred_prob_df,tmp],axis=1,ignore_index=True)


            pred_prob_df = pd.DataFrame(pred_prob_df.values,columns=mark_col_name)   
            plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei'] 
            plt.rcParams['axes.unicode_minus'] = False
            def plot_porb(flag):
                fig, ax = plt.subplots()
                ax = sns.barplot(x=pred_prob_df.T.index,y=1,data=pred_prob_df.T)
                ax.set_title('預測機率分布')
                plt.xticks(rotation=45)
                st.pyplot(fig)           
            plot_porb(1)
    if step == '2':
        if st.button("start Predict ",key=2):
            #load txt
            rule = []
            string = '(twitter)'
            f = open(path+'/Desktop/streamlit_app/sort 分析詞_1105.txt', encoding='UTF-8') 
            lines = f.readlines()
            for line in lines:
                line = line.strip('\n')
                string = string +'|'+ str(line)
            f.close()
            #feature
            feature = []
            a = re.findall(string,str(text), flags=re.IGNORECASE)
            s = ''
            for j in a :
                s = ''.join(j)+' '+s
            feature.append(s)
            # tfidf
            vectorizer2 = TfidfVectorizer(smooth_idf=True)
            #得到TF Matrix
            feature_empty = ['']
            if feature == feature_empty:
                fin_tfVec = pd.DataFrame(columns=feature_col_name)
                if fin_tfVec.empty:
                    fin_tfVec.loc[0] = 0
            else:
                tfidf = vectorizer2.fit_transform(feature)
            #词的对应索引值
                tf_feature_name = vectorizer2.get_feature_names_out() 
                tfVec = pd.DataFrame(tfidf.toarray(),columns=tf_feature_name)
                fin_tfVec = pd.DataFrame(data=tfVec,columns=feature_col_name)
                #st.write(fin_tfVec)
                fin_tfVec.replace(np.nan,0,inplace=True)
            pred = model.predict(fin_tfVec)
            pred_df = pd.DataFrame(pred,columns=mark_col_name)
            pred_prob = model.predict_proba(fin_tfVec)
            ###
            train_not_mark = pd.DataFrame(columns=mark_col_name)
            for i in range(pred.shape[0]):
                if pred_df.loc[i].sum() == 0 :
                    tmp = pd.DataFrame(pred_df.iloc[i])
                    train_not_mark = pd.concat([train_not_mark,tmp.T])
            pred_copy = pred.copy()
            for i in range(train_not_mark.shape[0]):
                a= pd.DataFrame(pred_df.loc[train_not_mark.index[i]])
                a=a.T
                prediction_of_probability = model.predict_proba(a)
                x = np.array(prediction_of_probability)[:,:,1]                
                for j in range(0,len(x)) :
                    if np.max(x) == x[j]:
                        pred_copy.loc[a.index[0]][j]=1
            ###
            importence = model.feature_importances_
            names = [fin_tfVec.columns[i] for i,j in enumerate(importence) if j != 0 ]
            model_import_features = pd.DataFrame()
            for i in range(len(names)):
                tmp = fin_tfVec[names[i]]
                model_import_features = pd.concat([model_import_features,tmp],axis = 1)
            model_equal0 = pd.DataFrame(columns=model_import_features.columns)
            model_not0 = pd.DataFrame(columns=model_import_features.columns)
            for i in range(model_import_features.shape[0]):
                tmp = model_import_features.iloc[i]
                t = tmp.sum()
                if t==0 :
                    tmp = pd.DataFrame(tmp).T
                    model_equal0 = pd.concat([model_equal0,tmp])
                else :
                    tmp = pd.DataFrame(tmp).T
                    model_not0 = pd.concat([model_not0,tmp])
            model_not0_df = fin_tfVec.loc[model_not0.index]
            model_equal0_df = fin_tfVec.loc[model_equal0.index]
            ###
            pred = model_1.predict(model_not0_df)
            pred_df = pd.DataFrame(pred,columns=mark_col_name)
            pred_prob = model_1.predict_proba(model_not0_df)
            ###
            st.subheader("查看各個標籤的機率值")
            pred_prob_df = pd.DataFrame()      
            for i in range(len(pred_prob)):
                tmp = pd.DataFrame(pred_prob[i]).T
                pred_prob_df=pd.concat([pred_prob_df,tmp],axis=1,ignore_index=True)

            pred_prob_df = pd.DataFrame(pred_prob_df.values,columns=mark_col_name)        
            plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei'] 
            plt.rcParams['axes.unicode_minus'] = False
            def plot_porb(flag):
                fig, ax = plt.subplots()
                ax = sns.barplot(x=pred_prob_df.T.index, y=1,data=pred_prob_df.T)
                ax.set_title('預測機率分布')
                plt.xticks(rotation=45)
                st.pyplot(fig)           
            plot_porb(1)
    if step == '3':
        if st.button("start Predict ",key=3):
            #load txt
            rule = []
            string = '(twitter)'
            f = open(path+'/Desktop/streamlit_app/sort 分析詞_1105.txt', encoding='UTF-8') 
            lines = f.readlines()
            for line in lines:
                line = line.strip('\n')
                string = string +'|'+ str(line)
            f.close()
            #feature
            feature = []
            a = re.findall(string,str(text), flags=re.IGNORECASE)
            s = ''
            for j in a :
                s = ''.join(j)+' '+s
            feature.append(s)
            # tfidf
            vectorizer2 = TfidfVectorizer(smooth_idf=True)
            #得到TF Matrix
            feature_empty = ['']
            if feature == feature_empty:
                fin_tfVec = pd.DataFrame(columns=feature_col_name)
                if fin_tfVec.empty:
                    fin_tfVec.loc[0] = 0
            else:
                tfidf = vectorizer2.fit_transform(feature)
            #词的对应索引值
                tf_feature_name = vectorizer2.get_feature_names_out() 
                tfVec = pd.DataFrame(tfidf.toarray(),columns=tf_feature_name)
                fin_tfVec = pd.DataFrame(data=tfVec,columns=feature_col_name)
                #st.write(fin_tfVec)
                fin_tfVec.replace(np.nan,0,inplace=True)
            pred = model.predict(fin_tfVec)
            pred_df = pd.DataFrame(pred,columns=mark_col_name)
            pred_prob = model.predict_proba(fin_tfVec)
            ###
            train_not_mark = pd.DataFrame(columns=mark_col_name)
            for i in range(pred.shape[0]):
                if pred_df.loc[i].sum() == 0 :
                    tmp = pd.DataFrame(pred_df.iloc[i])
                    train_not_mark = pd.concat([train_not_mark,tmp.T])
            pred_copy = pred.copy()
            for i in range(train_not_mark.shape[0]):
                a= pd.DataFrame(pred_df.loc[train_not_mark.index[i]])
                a=a.T
                prediction_of_probability = model.predict_proba(a)
                x = np.array(prediction_of_probability)[:,:,1]                
                for j in range(0,len(x)) :
                    if np.max(x) == x[j]:
                        pred_copy.loc[a.index[0]][j]=1
            ###
            importence = model.feature_importances_
            names = [fin_tfVec.columns[i] for i,j in enumerate(importence) if j != 0 ]
            model_import_features = pd.DataFrame()
            for i in range(len(names)):
                tmp = fin_tfVec[names[i]]
                model_import_features = pd.concat([model_import_features,tmp],axis = 1)
            model_equal0 = pd.DataFrame(columns=model_import_features.columns)
            model_not0 = pd.DataFrame(columns=model_import_features.columns)
            for i in range(model_import_features.shape[0]):
                tmp = model_import_features.iloc[i]
                t = tmp.sum()
                if t==0 :
                    tmp = pd.DataFrame(tmp).T
                    model_equal0 = pd.concat([model_equal0,tmp])
                else :
                    tmp = pd.DataFrame(tmp).T
                    model_not0 = pd.concat([model_not0,tmp])
            model_not0_df = fin_tfVec.loc[model_not0.index]
            model_equal0_df = fin_tfVec.loc[model_equal0.index]
            ###
            pred = model_1.predict(model_not0_df)
            pred_df = pd.DataFrame(pred,columns=mark_col_name)
            pred_prob = model_1.predict_proba(model_not0_df)
            ###
            importence = model_1.feature_importances_
            names = [fin_tfVec.columns[i] for i,j in enumerate(importence) if j != 0 ]
            model_1_import_features = pd.DataFrame()
            for i in range(len(names)):
                tmp = fin_tfVec[names[i]]
                model_1_import_features = pd.concat([model_1_import_features,tmp],axis = 1)
            model_1_equal0 = pd.DataFrame(columns=model_1_import_features.columns)
            model_1_not0 = pd.DataFrame(columns=model_1_import_features.columns)
            for i in range(model_1_import_features.shape[0]):
                tmp = model_1_import_features.iloc[i]
                t = tmp.sum()
                if t==0 :
                    tmp = pd.DataFrame(tmp).T
                    model_1_equal0 = pd.concat([model_1_equal0,tmp])
                else :
                    tmp = pd.DataFrame(tmp).T
                    model_1_not0 = pd.concat([model_1_not0,tmp])
            model_1_not0_df = fin_tfVec.loc[model_1_not0.index]
            model_1_equal0_df = fin_tfVec.loc[model_1_equal0.index]
            ###
            pred = model_2.predict(model_1_not0_df)
            pred_df = pd.DataFrame(pred,columns=mark_col_name)
            pred_prob = model_2.predict_proba(model_not0_df)
            ###
            st.subheader("查看各個標籤的機率值")
            pred_prob_df = pd.DataFrame()      
            for i in range(len(pred_prob)):
                tmp = pd.DataFrame(pred_prob[i]).T
                pred_prob_df=pd.concat([pred_prob_df,tmp],axis=1,ignore_index=True)

            pred_prob_df = pd.DataFrame(pred_prob_df.values,columns=mark_col_name)        
            plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei'] 
            plt.rcParams['axes.unicode_minus'] = False
            def plot_porb(flag):
                fig, ax = plt.subplots()
                ax = sns.barplot(x=pred_prob_df.T.index, y=1,data=pred_prob_df.T)
                ax.set_title('預測機率分布')
                plt.xticks(rotation=45)
                st.pyplot(fig)           
            plot_porb(1)