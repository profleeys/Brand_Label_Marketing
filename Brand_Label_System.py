from wordcloud import WordCloud, ImageColorGenerator
from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib.pyplot as plt
import streamlit as st
import numpy as np
from PIL import Image
import joblib
import re
import time
import pandas as pd
import os
#import dask.dataframe as pd

###
#            train_not_mark = pd.DataFrame(columns=mark_df_name)
#            for i in range(pred.shape[0]):
#                if pred_df.loc[i].sum() == 0 :
#                    tmp = pd.DataFrame(pred_df.iloc[i])
#                    train_not_mark = pd.concat([train_not_mark,tmp.T])
#            pred_copy = pred.copy()
#            for i in range(train_not_mark.shape[0]):
#                a= pd.DataFrame(pred_df.loc[train_not_mark.index[i]])
#                a=a.T
#                prediction_of_probability = model.predict_proba(a)
#                x = np.array(prediction_of_probability)[:,:,1]                
#                for j in range(0,len(x)) :
#                    if np.max(x) == x[j]:
#                        pred_copy.loc[a.index[0]][j]=1
###
def main_page():
    st.markdown("# Main page")
def page2():
    st.markdown("# Page 2 ")
def page3():
    st.markdown("# Page 3 ")
page_names_to_funcs = {"Main page": main_page,"Page 2": page2,"Page 3": page3}
###
@st.cache
def convert_df(df):
    return df.to_csv().encode('utf-8')
###
st.title('品牌標籤')
path = os.getcwd()
df_brand = pd.read_csv(path+'/Brand.csv',header=0)
df_nocut = pd.read_csv(path+'/not_cut.csv')
feature_df = pd.read_csv(path+'/Feature_small.csv')
mark_df = pd.read_csv(path+'/Mark_small.csv')
model = joblib.load(path+'/dt_1step_model.pkl')
model_1 = joblib.load(path+'/dt_2step_model.pkl')
model_2 = joblib.load(path+'/dt_3step_model.pkl')
###
list_1 = df_brand['Brand'].tolist()
list_2 = df_brand['Brand_1'].tolist()
list_3 = df_brand['Brand_2'].tolist()
list_4 = df_brand['Brand_3'].tolist()
list_5 = df_brand['Brand_4'].tolist()
dct = {}
for i, j in zip(list_1, list_1):
    dct.setdefault(i, []).append(j)
for i, j in zip(list_1, list_2):
    dct.setdefault(i, []).append(j)
for i, j in zip(list_1, list_3):
    dct.setdefault(i, []).append(j)
for i, j in zip(list_1, list_4):
    dct.setdefault(i, []).append(j)
for i, j in zip(list_1, list_5):
    dct.setdefault(i, []).append(j)
###
feature_col_name = feature_df.columns
mark_df_name = mark_df.columns
keys = mark_df_name.tolist()
model_names = ['DecisionTree(1step)','DecisionTree(2step)','DecisionTree(3step)']
###
labels = st.sidebar.multiselect('標籤(可多選)',keys,key=1)
keys_index = []
for i in range(len(labels)):
    keys_index.append(keys.index(labels[i]))
owners = st.sidebar.multiselect('品牌選擇(請選擇2個品牌)',df_brand['Brand'],key=2)
###
radio = st.sidebar.radio('模型選擇', model_names)
###
if radio == 'DecisionTree(1step)' :
    if len(owners) != 2:
        st.write(' ')
    else:
        result1 = st.sidebar.button('確定',key=4)
        if result1 :
            ###
            temp = dct.get(owners[0])
            temp1 = dct.get(owners[1])
            temp = '|'.join('%s' %id for id in temp)
            temp1 = '|'.join('%s' %id for id in temp1)
            ###
            brand_list = []
            brand_list1 = []
            for i in range(len(df_nocut)):
                temp_str = df_nocut['paragraph'][i]
                if (re.search((temp),str(temp_str))) != None:
                    brand_list.append(df_nocut['paragraph'][i])
                if (re.search((temp1),str(temp_str))) != None:
                    brand_list1.append(df_nocut['paragraph'][i])
            ###
            def cut_sent(para):
                para = re.sub('[， 。 ！ \？ ? ! ； \s]', '\n', str(para))  #no  Punctuation
                para = re.sub('[a-zA-z]+://[^\s]*jpg', "\n", str(para))#no  jpg
                para = re.sub('[a-zA-z]+://[^\s]*png', "\n", str(para))#no  png
                para = re.sub('[a-zA-z]+://[^\s]*[0-9]', "\n", str(para))#no  http
                para = re.sub('[a-zA-z]+://[^\s]*[a-zA-z]', "\n", str(para))#no  http
                para = re.sub('\:', "\n", str(para))#no
                para = para.rstrip()  # 段尾如果有多余的\n就去掉它
                return para.split("\n")
            ###    
            a=[]
            for i in range(len(brand_list)):
                pare = ''.join(brand_list[i].split())
                sent = cut_sent(pare)
                a.append(sent)
            b=[]
            for i in range(len(brand_list1)):
                pare = ''.join(brand_list1[i].split())
                sent = cut_sent(pare)
                b.append(sent)
            ###
            cut = pd.DataFrame(a)
            row = cut.shape[0]
            col = cut.shape[1]
            sent = []
            for i in range(0,col,2):
                for j in range (1,row):
                    n=(str(cut.iat[j,i]))
                    sent.append(n)
            ###
            cut1 = pd.DataFrame(b)
            row1 = cut1.shape[0]
            col1 = cut1.shape[1]
            sent1 = []
            for i in range(0,col1,2):
                for j in range (1,row1):
                    n=(str(cut1.iat[j,i]))
                    sent1.append(n)
            ###
            mark_dict = {'sent': sent}
            mark_dict1 = {'sent': sent1}
            df = pd.DataFrame(mark_dict)
            df1 = pd.DataFrame(mark_dict1)
            ###
            df.replace('None',np.NaN,inplace=True)
            df.replace('',np.NaN,inplace=True)
            df.dropna(axis=0,inplace=True)
            df.reset_index(drop=True, inplace=True)
            df_new = pd.DataFrame(df)
            df1.replace('None',np.NaN,inplace=True)
            df1.replace('',np.NaN,inplace=True)
            df1.dropna(axis=0,inplace=True)
            df1.reset_index(drop=True, inplace=True)
            df_new1 = pd.DataFrame(df1)
            ###
            string = '(twitter)'
            all=set()
            f = open(path+'/sort 分析詞_1105.txt', encoding='UTF-8')
            lines = f.readlines()
            for line in lines:
                line = line.strip('\n')
                string = string +'|'+ str(line)
            f.close()
            ###
            feature = []
            for i in range (len(df_new['sent'])):
                a = re.findall(string,str(df_new["sent"][i]), flags=re.IGNORECASE)
                s = ''
                for j in a :
                    s = ''.join(j)+' '+s
                feature.append(s)
            feature1 = []
            for i in range (len(df_new1['sent'])):
                a = re.findall(string,str(df_new1["sent"][i]), flags=re.IGNORECASE)
                s = ''
                for j in a :
                    s = ''.join(j)+' '+s
                feature1.append(s)
            ###
            vectorizer2 = TfidfVectorizer(smooth_idf=True)
            if feature == []:
                fin_tfVec_loc = pd.DataFrame(columns=feature_col_name)
                dict_list = []
                pass
            else:
                tfidf = vectorizer2.fit_transform(feature)
                tf_feature_name = vectorizer2.get_feature_names_out()
                tfVec = pd.DataFrame(tfidf.toarray(),columns=tf_feature_name)
                fin_tfVec = pd.DataFrame(columns=feature_col_name)
                for i in range(len(tfVec.columns)):
                    for j in range(len(fin_tfVec.columns)):
                        if  tfVec.columns[i] == fin_tfVec.columns[j]:
                            fin_tfVec[fin_tfVec.columns[j]] = tfVec[fin_tfVec.columns[j]]
                fin_tfVec.fillna(value=0, method=None, axis=1, inplace=True, limit=None, downcast=None)
                pred = model.predict(fin_tfVec)
                pred_df = pd.DataFrame(data=pred,columns=mark_df_name)
                label = []
                sent_loc = []
                fin_tfVec_loc = pd.DataFrame(columns=feature_col_name)
                temp = pd.DataFrame(columns=feature_col_name)#暫存
                for i in range(len(labels)):
                    label.append(pred_df[pred_df[mark_df_name[keys_index[i]]] > 0].index)
                for j in range(len(labels)):
                    temp = fin_tfVec.loc[label[j]]                
                    if temp.empty:     
                        pass
                    else:
                        fin_tfVec_loc = pd.concat([fin_tfVec_loc,temp])
                temp = list(fin_tfVec_loc.index)
                for i in range(len(fin_tfVec_loc.index)):
                    sent_loc.append(df_new.loc[temp[i]])
                sent_loc = pd.DataFrame(data=sent_loc)
                csv = convert_df(sent_loc)
                n = fin_tfVec_loc.sum(axis = 0)
                m = pd.DataFrame(n)
                m.columns = ['sum']
                o = m.drop(m[m['sum'] == 0].index)
                p = m.drop(m[m['sum'] == 0].index).T
                p.columns.tolist()
                dict_list = []
                keys = p.columns.tolist() # 列A
                values = o['sum'].tolist() # 列B
                dict_list = dict(zip(keys, values))
            ###
            vectorizer2 = TfidfVectorizer(smooth_idf=True)
            if feature1 == []:
                fin_tfVec1_loc = pd.DataFrame(columns=feature_col_name) 
                dict_list1 = []
                pass
            else:
                tfidf1 = vectorizer2.fit_transform(feature1)
                tf_feature_name1 = vectorizer2.get_feature_names_out()
                tfVec1 = pd.DataFrame(tfidf1.toarray(),columns=tf_feature_name1)
                fin_tfVec1 = pd.DataFrame(columns=feature_col_name)
                for i in range(len(tfVec1.columns)):
                    for j in range(len(fin_tfVec1.columns)):
                        if  tfVec1.columns[i] == fin_tfVec1.columns[j]:
                            fin_tfVec1[fin_tfVec1.columns[j]] = tfVec1[fin_tfVec1.columns[j]]
                fin_tfVec1.fillna(value=0, method=None, axis=1, inplace=True, limit=None, downcast=None)
                pred = model.predict(fin_tfVec1)
                pred_df = pd.DataFrame(data=pred,columns=mark_df_name)
                label1 = []
                sent1_loc = []
                fin_tfVec1_loc = pd.DataFrame(columns=feature_col_name)    
                for i in range(len(labels)):
                    label1.append(pred_df[pred_df[mark_df_name[keys_index[i]]] >0 ].index)
                for j in range(len(labels)):
                    temp = fin_tfVec1.loc[label1[j]]
                    if temp.empty:     
                        pass
                    else:
                        fin_tfVec1_loc = pd.concat([fin_tfVec1_loc,temp]) 
                temp = list(fin_tfVec1_loc.index)
                for i in range(len(fin_tfVec1_loc.index)):
                    sent1_loc.append(df_new1.loc[temp[i]])
                sent1_loc = pd.DataFrame(data=sent1_loc)
                csv1 = convert_df(sent1_loc)
                q = fin_tfVec1_loc.sum(axis = 0)
                r = pd.DataFrame(q)
                r.columns = ['sum']
                s = r.drop(r[r['sum'] == 0].index)
                t = r.drop(r[r['sum'] == 0].index).T
                t.columns.tolist()
                dict_list1 = []
                keys = t.columns.tolist() # 列A
                values = s['sum'].tolist() # 列B
                dict_list1 = dict(zip(keys, values))
            ###
            if fin_tfVec_loc.empty & fin_tfVec1_loc.empty:
                dict_list2 = []
                pass
            else:   
                fin_tfVec2 = pd.concat([fin_tfVec_loc, fin_tfVec1_loc])            
                u = fin_tfVec2.sum(axis = 0)
                v = pd.DataFrame(u)
                v.columns = ['sum']
                w = v.drop(v[v['sum'] == 0].index)
                x = v.drop(v[v['sum'] == 0].index).T
                x.columns.tolist()
                dict_list2 = []
                keys = x.columns.tolist() # 列A
                values = w['sum'].tolist() # 列B
                dict_list2 = dict(zip(keys, values))
            ###
            st.set_option('deprecation.showPyplotGlobalUse', False)
            mask_image = np.array(Image.open(path+'/cloud.jpg'))
            ###
            if len(dict_list) == 0 :
                st.write(owners[0])
                st.write('無相關資訊')
            else:
                st.write(owners[0])
                wordcloud = WordCloud(background_color="white", font_path=path+'/msjh.ttf', max_words = 20, collocations=False, margin=2,mask=mask_image)
                wordcloud.generate_from_frequencies(dict_list)
                plt.figure(figsize=(20,10), facecolor='k')
                plt.imshow(wordcloud)
                plt.axis("off")
                st.pyplot()
                st.download_button(owners[0]+'文本下載', data=csv, file_name=owners[0]+'文本.csv')
            if len(dict_list1) == 0 :
                st.write(owners[1])
                st.write('無相關資訊')      
            else:      
                st.write(owners[1])
                wordcloud1 = WordCloud(background_color="white", font_path=path+'/msjh.ttf', max_words = 20, collocations=False, margin=2,mask=mask_image)
                wordcloud1.generate_from_frequencies(dict_list1)
                plt.figure(figsize=(20,10), facecolor='k')
                plt.imshow(wordcloud1)
                plt.axis("off")
                st.pyplot()
                st.download_button(owners[1]+'文本下載', data=csv1, file_name=owners[1]+'文本.csv')
            if len(dict_list2) == 0 :
                st.write(owners[0],'+',owners[1])
                st.write('無相關資訊')      
            else:    
                st.write(owners[0],'+',owners[1])
                wordcloud2 = WordCloud(background_color="white", font_path=path+'/msjh.ttf', max_words = 20, collocations=False, margin=2,mask=mask_image)
                wordcloud2.generate_from_frequencies(dict_list2)
                plt.figure(figsize=(20,10), facecolor='k')
                plt.imshow(wordcloud2)
                plt.axis("off")
                st.pyplot()
            ###
elif radio == 'DecisionTree(2step)' :
    if len(owners) != 2:
        st.write(' ')        
    else:
        result2 = st.sidebar.button('確定',key=5)
        if result2 :
            ###
            temp = dct.get(owners[0])
            temp1 = dct.get(owners[1])
            temp = '|'.join('%s' %id for id in temp)
            temp1 = '|'.join('%s' %id for id in temp1)
            ###
            brand_list = []
            brand_list1 = []
            for i in range(len(df_nocut)):
                temp_str = df_nocut['paragraph'][i]
                if (re.search((temp),str(temp_str))) != None:
                    brand_list.append(df_nocut['paragraph'][i])
                if (re.search((temp1),str(temp_str))) != None:
                    brand_list1.append(df_nocut['paragraph'][i])
            ###
            def cut_sent(para):
                para = re.sub('[， 。 ！ \？ ? ! ； \s]', '\n', str(para))  #no  Punctuation
                para = re.sub('[a-zA-z]+://[^\s]*jpg', "\n", str(para))#no  jpg
                para = re.sub('[a-zA-z]+://[^\s]*png', "\n", str(para))#no  png
                para = re.sub('[a-zA-z]+://[^\s]*[0-9]', "\n", str(para))#no  http
                para = re.sub('[a-zA-z]+://[^\s]*[a-zA-z]', "\n", str(para))#no  http
                para = re.sub('\:', "\n", str(para))#no
                para = para.rstrip()  # 段尾如果有多余的\n就去掉它
                return para.split("\n")
            ###    
            a=[]
            for i in range(len(brand_list)):
                pare = ''.join(brand_list[i].split())
                sent = cut_sent(pare)
                a.append(sent)
            b=[]
            for i in range(len(brand_list1)):
                pare = ''.join(brand_list1[i].split())
                sent = cut_sent(pare)
                b.append(sent)
            ###
            cut = pd.DataFrame(a)
            row = cut.shape[0]
            col = cut.shape[1]
            sent = []
            for i in range(0,col,2):
                for j in range (1,row):
                    n=(str(cut.iat[j,i]))
                    sent.append(n)
            ###
            cut1 = pd.DataFrame(b)
            row1 = cut1.shape[0]
            col1 = cut1.shape[1]
            sent1 = []
            for i in range(0,col1,2):
                for j in range (1,row1):
                    n=(str(cut1.iat[j,i]))
                    sent1.append(n)
            ###
            mark_dict = {'sent': sent}
            mark_dict1 = {'sent': sent1}
            df = pd.DataFrame(mark_dict)
            df1 = pd.DataFrame(mark_dict1)
            ###
            df.replace('None',np.NaN,inplace=True)
            df.replace('',np.NaN,inplace=True)
            df.dropna(axis=0,inplace=True)
            df.reset_index(drop=True, inplace=True)
            df_new = pd.DataFrame(df)
            df1.replace('None',np.NaN,inplace=True)
            df1.replace('',np.NaN,inplace=True)
            df1.dropna(axis=0,inplace=True)
            df1.reset_index(drop=True, inplace=True)
            df_new1 = pd.DataFrame(df1)
            ###
            string = '(twitter)'
            all=set()
            f = open(path+'/sort 分析詞_1105.txt', encoding='UTF-8')
            lines = f.readlines()
            for line in lines:
                line = line.strip('\n')
                string = string +'|'+ str(line)
            f.close()
            ###
            feature = []
            for i in range (len(df_new['sent'])):
                a = re.findall(string,str(df_new["sent"][i]), flags=re.IGNORECASE)
                s = ''
                for j in a :
                    s = ''.join(j)+' '+s
                feature.append(s)
            feature1 = []
            for i in range (len(df_new1['sent'])):
                a = re.findall(string,str(df_new1["sent"][i]), flags=re.IGNORECASE)
                s = ''
                for j in a :
                    s = ''.join(j)+' '+s
                feature1.append(s)
            ###
            vectorizer2 = TfidfVectorizer(smooth_idf=True)
            if feature == []:
                fin_tfVec_loc = pd.DataFrame(columns=feature_col_name)
                dict_list = []
                pass
            else:
                tfidf = vectorizer2.fit_transform(feature)
                tf_feature_name = vectorizer2.get_feature_names_out()
                tfVec = pd.DataFrame(tfidf.toarray(),columns=tf_feature_name)
                fin_tfVec = pd.DataFrame(columns=feature_col_name)
                for i in range(len(tfVec.columns)):
                    for j in range(len(fin_tfVec.columns)):
                        if  tfVec.columns[i] == fin_tfVec.columns[j]:
                            fin_tfVec[fin_tfVec.columns[j]] = tfVec[fin_tfVec.columns[j]]
                fin_tfVec.fillna(value=0, method=None, axis=1, inplace=True, limit=None, downcast=None)
                pred = model.predict(fin_tfVec)
                pred_df = pd.DataFrame(data=pred,columns=mark_df_name)
                ###
                label = []
                sent_loc = []
                fin_tfVec_loc = pd.DataFrame(columns=feature_col_name)
                temp = pd.DataFrame(columns=feature_col_name)#暫存
                for i in range(len(labels)):
                    label.append(pred_df[pred_df[mark_df_name[keys_index[i]]] > 0].index)
                for j in range(len(labels)):
                    temp = fin_tfVec.loc[label[j]]                
                    if temp.empty:     
                        pass
                    else:
                        fin_tfVec_loc = pd.concat([fin_tfVec_loc,temp])
                temp = list(fin_tfVec_loc.index)
                for i in range(len(fin_tfVec_loc.index)):
                    sent_loc.append(df_new.loc[temp[i]])
                sent_loc = pd.DataFrame(data=sent_loc)
                n = fin_tfVec_loc.sum(axis = 0)
                m = pd.DataFrame(n)
                m.columns = ['sum']
                o = m.drop(m[m['sum'] == 0].index)
                p = m.drop(m[m['sum'] == 0].index).T
                p.columns.tolist()
                dict_list = []
                keys = p.columns.tolist() # 列A
                values = o['sum'].tolist() # 列B
                dict_list = dict(zip(keys, values))
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
                pred = model_1.predict(model_not0_df)
                pred_df = pd.DataFrame(data=pred,columns=mark_df_name)
                ###
                label = []
                sent_loc_0 = []
                fin_tfVec_loc = pd.DataFrame(columns=feature_col_name)
                temp = pd.DataFrame(columns=feature_col_name)#暫存
                for i in range(len(labels)):
                    label.append(pred_df[pred_df[mark_df_name[keys_index[i]]] > 0].index)
                for j in range(len(labels)):
                    temp = fin_tfVec.loc[label[j]]                
                    if temp.empty:     
                        pass
                    else:
                        fin_tfVec_loc = pd.concat([fin_tfVec_loc,temp])
                temp = list(fin_tfVec_loc.index)
                for i in range(len(fin_tfVec_loc.index)):
                    sent_loc_0.append(df_new.loc[temp[i]])
                sent_loc_0= pd.DataFrame(data=sent_loc_0)
                sent_loc_concat = pd.concat([sent_loc,sent_loc_0]) 
                csv = convert_df(sent_loc_concat)
                n = fin_tfVec_loc.sum(axis = 0)
                m = pd.DataFrame(n)
                m.columns = ['sum']
                o = m.drop(m[m['sum'] == 0].index)
                p = m.drop(m[m['sum'] == 0].index).T
                p.columns.tolist()
                dict_list_0 = []
                keys = p.columns.tolist() # 列A
                values = o['sum'].tolist() # 列B
                dict_list_0 = dict(zip(keys, values))
                dict_list_concat={}
                dict_list_concat.update(dict_list)
                dict_list_concat.update(dict_list_0)
            ###
            vectorizer2 = TfidfVectorizer(smooth_idf=True)
            if feature1 == []:
                fin_tfVec1_loc = pd.DataFrame(columns=feature_col_name) 
                dict_list1 = []
                pass
            else:
                tfidf1 = vectorizer2.fit_transform(feature1)
                tf_feature_name1 = vectorizer2.get_feature_names_out()
                tfVec1 = pd.DataFrame(tfidf1.toarray(),columns=tf_feature_name1)
                fin_tfVec1 = pd.DataFrame(columns=feature_col_name)
                for i in range(len(tfVec1.columns)):
                    for j in range(len(fin_tfVec1.columns)):
                        if  tfVec1.columns[i] == fin_tfVec1.columns[j]:
                            fin_tfVec1[fin_tfVec1.columns[j]] = tfVec1[fin_tfVec1.columns[j]]
                fin_tfVec1.fillna(value=0, method=None, axis=1, inplace=True, limit=None, downcast=None)
                pred = model.predict(fin_tfVec1)
                pred_df = pd.DataFrame(data=pred,columns=mark_df_name)
                ###
                label1 = []
                sent1_loc = []
                fin_tfVec1_loc = pd.DataFrame(columns=feature_col_name)    
                for i in range(len(labels)):
                    label1.append(pred_df[pred_df[mark_df_name[keys_index[i]]] >0 ].index)
                for j in range(len(labels)):
                    temp = fin_tfVec1.loc[label1[j]]
                    if temp.empty:     
                        pass
                    else:
                        fin_tfVec1_loc = pd.concat([fin_tfVec1_loc,temp]) 
                temp = list(fin_tfVec1_loc.index)
                for i in range(len(fin_tfVec1_loc.index)):
                    sent1_loc.append(df_new1.loc[temp[i]])
                sent1_loc = pd.DataFrame(data=sent1_loc)
                q = fin_tfVec1_loc.sum(axis = 0)
                r = pd.DataFrame(q)
                r.columns = ['sum']
                s = r.drop(r[r['sum'] == 0].index)
                t = r.drop(r[r['sum'] == 0].index).T
                t.columns.tolist()
                dict_list1 = []
                keys = t.columns.tolist() # 列A
                values = s['sum'].tolist() # 列B
                dict_list1 = dict(zip(keys, values))
                ###
                importence = model.feature_importances_
                names = [fin_tfVec1.columns[i] for i,j in enumerate(importence) if j != 0 ]
                model_import_features = pd.DataFrame()
                for i in range(len(names)):
                    tmp = fin_tfVec1[names[i]]
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
                model_not0_df = fin_tfVec1.loc[model_not0.index]
                model_equal0_df = fin_tfVec1.loc[model_equal0.index]
                pred = model_1.predict(fin_tfVec1)
                pred_df = pd.DataFrame(data=pred,columns=mark_df_name)                
                ###
                label1 = []
                sent1_loc_0 = []
                fin_tfVec1_loc = pd.DataFrame(columns=feature_col_name)    
                for i in range(len(labels)):
                    label1.append(pred_df[pred_df[mark_df_name[keys_index[i]]] >0 ].index)
                for j in range(len(labels)):
                    temp = fin_tfVec1.loc[label1[j]]
                    if temp.empty:     
                        pass
                    else:
                        fin_tfVec1_loc = pd.concat([fin_tfVec1_loc,temp]) 
                temp = list(fin_tfVec1_loc.index)
                for i in range(len(fin_tfVec1_loc.index)):
                    sent1_loc_0.append(df_new1.loc[temp[i]])
                sent1_loc_0 = pd.DataFrame(data=sent1_loc_0)
                sent1_loc_concat = pd.concat([sent1_loc,sent1_loc_0]) 
                csv1 = convert_df(sent1_loc_concat)
                q = fin_tfVec1_loc.sum(axis = 0)
                r = pd.DataFrame(q)
                r.columns = ['sum']
                s = r.drop(r[r['sum'] == 0].index)
                t = r.drop(r[r['sum'] == 0].index).T
                t.columns.tolist()
                dict_list1_0 = []
                keys = t.columns.tolist() # 列A
                values = s['sum'].tolist() # 列B
                dict_list1_0 = dict(zip(keys, values))
                dict_list1_concat={}
                dict_list1_concat.update(dict_list1)
                dict_list1_concat.update(dict_list1_0)
            ###
            if fin_tfVec_loc.empty & fin_tfVec1_loc.empty:
                dict_list2_concat={}
                pass
            else:   
                dict_list2_concat={}
                dict_list2_concat.update(dict_list_concat)
                dict_list2_concat.update(dict_list1_concat)
            ###
            st.set_option('deprecation.showPyplotGlobalUse', False)
            mask_image = np.array(Image.open(path+'/cloud.jpg'))
            ###
            if len(dict_list) == 0 :
                st.write(owners[0])
                st.write('無相關資訊')
            else:
                st.write(owners[0])
                wordcloud = WordCloud(background_color="white", font_path=path+'/msjh.ttf', max_words = 20, collocations=False, margin=2,mask=mask_image)
                wordcloud.generate_from_frequencies(dict_list_concat)
                plt.figure(figsize=(20,10), facecolor='k')
                plt.imshow(wordcloud)
                plt.axis("off")
                st.pyplot()
                st.download_button(owners[0]+'文本下載', data=csv, file_name=owners[0]+'文本.csv')
            if len(dict_list1) == 0 :
                st.write(owners[1])
                st.write('無相關資訊')      
            else:      
                st.write(owners[1])
                wordcloud1 = WordCloud(background_color="white", font_path=path+'/msjh.ttf', max_words = 20, collocations=False, margin=2,mask=mask_image)
                wordcloud1.generate_from_frequencies(dict_list1_concat)
                plt.figure(figsize=(20,10), facecolor='k')
                plt.imshow(wordcloud1)
                plt.axis("off")
                st.pyplot()
                st.download_button(owners[1]+'文本下載', data=csv1, file_name=owners[1]+'文本.csv')
            if len(dict_list)|len(dict_list1) == 0 :
                st.write(owners[0],'+',owners[1])
                st.write('無相關資訊')      
            else:    
                st.write(owners[0],'+',owners[1])
                wordcloud2 = WordCloud(background_color="white", font_path=path+'/msjh.ttf', max_words = 20, collocations=False, margin=2,mask=mask_image)
                wordcloud2.generate_from_frequencies(dict_list2_concat)
                plt.figure(figsize=(20,10), facecolor='k')
                plt.imshow(wordcloud2)
                plt.axis("off")
                st.pyplot()
            ###
elif radio == 'DecisionTree(3step)' :
    if len(owners) != 2:
        st.write(' ')
    else:
        result3 = st.sidebar.button('確定',key=6)
        if result3 :
            temp = dct.get(owners[0])
            temp1 = dct.get(owners[1])
            temp = '|'.join('%s' %id for id in temp)
            temp1 = '|'.join('%s' %id for id in temp1)
            #st.write(temp)
            #st.write(temp1)
            ###
            brand_list = []
            brand_list1 = []
            for i in range(len(df_nocut)):
                temp_str = df_nocut['paragraph'][i]
                if (re.search((temp),str(temp_str))) != None:
                    brand_list.append(df_nocut['paragraph'][i])
                if (re.search((temp1),str(temp_str))) != None:
                    brand_list1.append(df_nocut['paragraph'][i])
            #st.write(brand_list)
            #st.write(brand_list1)
            ###
            def cut_sent(para):
                para = re.sub('[， 。 ！ \？ ? ! ； \s]', '\n', str(para))  #no  Punctuation
                para = re.sub('[a-zA-z]+://[^\s]*jpg', "\n", str(para))#no  jpg
                para = re.sub('[a-zA-z]+://[^\s]*png', "\n", str(para))#no  png
                para = re.sub('[a-zA-z]+://[^\s]*[0-9]', "\n", str(para))#no  http
                para = re.sub('[a-zA-z]+://[^\s]*[a-zA-z]', "\n", str(para))#no  http
                para = re.sub('\:', "\n", str(para))#no
                para = para.rstrip()  # 段尾如果有多余的\n就去掉它
                return para.split("\n")
            ###    
            a=[]
            for i in range(len(brand_list)):
                pare = ''.join(brand_list[i].split())
                sent = cut_sent(pare)
                a.append(sent)
            b=[]
            for i in range(len(brand_list1)):
                pare = ''.join(brand_list1[i].split())
                sent = cut_sent(pare)
                b.append(sent)
            ###
            cut = pd.DataFrame(a)
            row = cut.shape[0]
            col = cut.shape[1]
            sent = []
            for i in range(0,col,2):
                for j in range (1,row):
                    n=(str(cut.iat[j,i]))
                    sent.append(n)
            ###
            cut1 = pd.DataFrame(b)
            row1 = cut1.shape[0]
            col1 = cut1.shape[1]
            sent1 = []
            for i in range(0,col1,2):
                for j in range (1,row1):
                    n=(str(cut1.iat[j,i]))
                    sent1.append(n)
            ###
            mark_dict = {'sent': sent}
            mark_dict1 = {'sent': sent1}
            df = pd.DataFrame(mark_dict)
            df1 = pd.DataFrame(mark_dict1)
            ###
            df.replace('None',np.NaN,inplace=True)
            df.replace('',np.NaN,inplace=True)
            df.dropna(axis=0,inplace=True)
            df.reset_index(drop=True, inplace=True)
            df_new = pd.DataFrame(df)
            df1.replace('None',np.NaN,inplace=True)
            df1.replace('',np.NaN,inplace=True)
            df1.dropna(axis=0,inplace=True)
            df1.reset_index(drop=True, inplace=True)
            df_new1 = pd.DataFrame(df1)
            ###
            string = '(twitter)'
            all=set()
            f = open(path+'/sort 分析詞_1105.txt', encoding='UTF-8')
            lines = f.readlines()
            for line in lines:
                line = line.strip('\n')
                string = string +'|'+ str(line)
            f.close()
            ###
            feature = []
            for i in range (len(df_new['sent'])):
                a = re.findall(string,str(df_new["sent"][i]), flags=re.IGNORECASE)
                s = ''
                for j in a :
                    s = ''.join(j)+' '+s
                feature.append(s)
            feature1 = []
            for i in range (len(df_new1['sent'])):
                a = re.findall(string,str(df_new1["sent"][i]), flags=re.IGNORECASE)
                s = ''
                for j in a :
                    s = ''.join(j)+' '+s
                feature1.append(s)
            ###
            vectorizer2 = TfidfVectorizer(smooth_idf=True)
            if feature == []:
                fin_tfVec_loc = pd.DataFrame(columns=feature_col_name)
                dict_list = []
                pass
            else:
                tfidf = vectorizer2.fit_transform(feature)
                tf_feature_name = vectorizer2.get_feature_names_out()
                tfVec = pd.DataFrame(tfidf.toarray(),columns=tf_feature_name)
                fin_tfVec = pd.DataFrame(columns=feature_col_name)
                for i in range(len(tfVec.columns)):
                    for j in range(len(fin_tfVec.columns)):
                        if  tfVec.columns[i] == fin_tfVec.columns[j]:
                            fin_tfVec[fin_tfVec.columns[j]] = tfVec[fin_tfVec.columns[j]]
                fin_tfVec.fillna(value=0, method=None, axis=1, inplace=True, limit=None, downcast=None)
                pred = model.predict(fin_tfVec)
                pred_df = pd.DataFrame(data=pred,columns=mark_df_name)
                ###
                label = []
                sent_loc = []
                fin_tfVec_loc = pd.DataFrame(columns=feature_col_name)
                temp = pd.DataFrame(columns=feature_col_name)#暫存
                for i in range(len(labels)):
                    label.append(pred_df[pred_df[mark_df_name[keys_index[i]]] > 0].index)
                for j in range(len(labels)):
                    temp = fin_tfVec.loc[label[j]]                
                    if temp.empty:     
                        pass
                    else:
                        fin_tfVec_loc = pd.concat([fin_tfVec_loc,temp])
                temp = list(fin_tfVec_loc.index)
                for i in range(len(fin_tfVec_loc.index)):
                    sent_loc.append(df_new.loc[temp[i]])
                sent_loc = pd.DataFrame(data=sent_loc)
                n = fin_tfVec_loc.sum(axis = 0)
                m = pd.DataFrame(n)
                m.columns = ['sum']
                o = m.drop(m[m['sum'] == 0].index)
                p = m.drop(m[m['sum'] == 0].index).T
                p.columns.tolist()
                dict_list = []
                keys = p.columns.tolist() # 列A
                values = o['sum'].tolist() # 列B
                dict_list = dict(zip(keys, values))
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
                pred = model_1.predict(model_not0_df)
                pred_df = pd.DataFrame(data=pred,columns=mark_df_name)
                ###
                label = []
                sent_loc_0 = []
                fin_tfVec_loc = pd.DataFrame(columns=feature_col_name)
                temp = pd.DataFrame(columns=feature_col_name)#暫存
                for i in range(len(labels)):
                    label.append(pred_df[pred_df[mark_df_name[keys_index[i]]] > 0].index)
                for j in range(len(labels)):
                    temp = fin_tfVec.loc[label[j]]                
                    if temp.empty:     
                        pass
                    else:
                        fin_tfVec_loc = pd.concat([fin_tfVec_loc,temp])
                temp = list(fin_tfVec_loc.index)
                for i in range(len(fin_tfVec_loc.index)):
                    sent_loc_0.append(df_new.loc[temp[i]])
                sent_loc_0 = pd.DataFrame(data=sent_loc_0)
                sent_loc_concat = pd.concat([sent_loc,sent_loc_0]) 
                n = fin_tfVec_loc.sum(axis = 0)
                m = pd.DataFrame(n)
                m.columns = ['sum']
                o = m.drop(m[m['sum'] == 0].index)
                p = m.drop(m[m['sum'] == 0].index).T
                p.columns.tolist()
                dict_list_0 = []
                keys = p.columns.tolist() # 列A
                values = o['sum'].tolist() # 列B
                dict_list_0 = dict(zip(keys, values))
                dict_list_concat={}
                dict_list_concat.update(dict_list)
                dict_list_concat.update(dict_list_0)
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
                pred_df = pd.DataFrame(data=pred,columns=mark_df_name)
                ###
                label = []
                sent_loc_1 = []
                fin_tfVec_loc = pd.DataFrame(columns=feature_col_name)
                temp = pd.DataFrame(columns=feature_col_name)#暫存
                for i in range(len(labels)):
                    label.append(pred_df[pred_df[mark_df_name[keys_index[i]]] > 0].index)
                for j in range(len(labels)):
                    temp = fin_tfVec.loc[label[j]]                
                    if temp.empty:     
                        pass
                    else:
                        fin_tfVec_loc = pd.concat([fin_tfVec_loc,temp])
                temp = list(fin_tfVec_loc.index)
                for i in range(len(fin_tfVec_loc.index)):
                    sent_loc_1.append(df_new.loc[temp[i]])
                sent_loc_1 = pd.DataFrame(data=sent_loc_1)
                sent_loc_concat = pd.concat([sent_loc_concat,sent_loc_1]) 
                csv = convert_df(sent_loc_concat)
                n = fin_tfVec_loc.sum(axis = 0)
                m = pd.DataFrame(n)
                m.columns = ['sum']
                o = m.drop(m[m['sum'] == 0].index)
                p = m.drop(m[m['sum'] == 0].index).T
                p.columns.tolist()
                dict_list_1 = []
                keys = p.columns.tolist() # 列A
                values = o['sum'].tolist() # 列B
                dict_list_1 = dict(zip(keys, values))
                dict_list_concat.update(dict_list_1)
            ###
            vectorizer2 = TfidfVectorizer(smooth_idf=True)
            if feature1 == []:
                fin_tfVec1_loc = pd.DataFrame(columns=feature_col_name) 
                dict_list1 = []
                pass
            else:
                tfidf1 = vectorizer2.fit_transform(feature1)
                tf_feature_name1 = vectorizer2.get_feature_names_out()
                tfVec1 = pd.DataFrame(tfidf1.toarray(),columns=tf_feature_name1)
                fin_tfVec1 = pd.DataFrame(columns=feature_col_name)
                for i in range(len(tfVec1.columns)):
                    for j in range(len(fin_tfVec1.columns)):
                        if  tfVec1.columns[i] == fin_tfVec1.columns[j]:
                            fin_tfVec1[fin_tfVec1.columns[j]] = tfVec1[fin_tfVec1.columns[j]]
                fin_tfVec1.fillna(value=0, method=None, axis=1, inplace=True, limit=None, downcast=None)
                pred = model.predict(fin_tfVec1)
                pred_df = pd.DataFrame(data=pred,columns=mark_df_name)
                ###
                label1 = []
                sent1_loc = []
                fin_tfVec1_loc = pd.DataFrame(columns=feature_col_name)    
                for i in range(len(labels)):
                    label1.append(pred_df[pred_df[mark_df_name[keys_index[i]]] >0 ].index)
                for j in range(len(labels)):
                    temp = fin_tfVec1.loc[label1[j]]
                    if temp.empty:     
                        pass
                    else:
                        fin_tfVec1_loc = pd.concat([fin_tfVec1_loc,temp]) 
                temp = list(fin_tfVec1_loc.index)
                for i in range(len(fin_tfVec1_loc.index)):
                    sent1_loc.append(df_new1.loc[temp[i]])
                sent1_loc = pd.DataFrame(data=sent1_loc)
                q = fin_tfVec1_loc.sum(axis = 0)
                r = pd.DataFrame(q)
                r.columns = ['sum']
                s = r.drop(r[r['sum'] == 0].index)
                t = r.drop(r[r['sum'] == 0].index).T
                t.columns.tolist()
                dict_list1 = []
                keys = t.columns.tolist() # 列A
                values = s['sum'].tolist() # 列B
                dict_list1 = dict(zip(keys, values))              
                ###
                importence = model.feature_importances_
                names = [fin_tfVec1.columns[i] for i,j in enumerate(importence) if j != 0 ]
                model_import_features = pd.DataFrame()
                for i in range(len(names)):
                    tmp = fin_tfVec1[names[i]]
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
                model_not0_df = fin_tfVec1.loc[model_not0.index]
                model_equal0_df = fin_tfVec1.loc[model_equal0.index]
                pred = model_1.predict(fin_tfVec1)
                pred_df = pd.DataFrame(data=pred,columns=mark_df_name)  
                ###
                label1 = []
                sent1_loc_0 = []
                fin_tfVec1_loc = pd.DataFrame(columns=feature_col_name)    
                for i in range(len(labels)):
                    label1.append(pred_df[pred_df[mark_df_name[keys_index[i]]] >0 ].index)
                for j in range(len(labels)):
                    temp = fin_tfVec1.loc[label1[j]]
                    if temp.empty:     
                        pass
                    else:
                        fin_tfVec1_loc = pd.concat([fin_tfVec1_loc,temp]) 
                temp = list(fin_tfVec1_loc.index)
                for i in range(len(fin_tfVec1_loc.index)):
                    sent1_loc_0.append(df_new1.loc[temp[i]])
                sent1_loc_0 = pd.DataFrame(data=sent1_loc_0)
                sent1_loc_concat = pd.concat([sent1_loc,sent1_loc_0]) 
                q = fin_tfVec1_loc.sum(axis = 0)
                r = pd.DataFrame(q)
                r.columns = ['sum']
                s = r.drop(r[r['sum'] == 0].index)
                t = r.drop(r[r['sum'] == 0].index).T
                t.columns.tolist()
                dict_list1_0 = []
                keys = t.columns.tolist() # 列A
                values = s['sum'].tolist() # 列B
                dict_list1_0 = dict(zip(keys, values))
                dict_list1_concat={}
                dict_list1_concat.update(dict_list1)
                dict_list1_concat.update(dict_list1_0)
                ###           
                importence = model_1.feature_importances_
                names = [fin_tfVec1.columns[i] for i,j in enumerate(importence) if j != 0 ]
                model_1_import_features = pd.DataFrame()
                for i in range(len(names)):
                    tmp = fin_tfVec1[names[i]]
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
                model_1_not0_df = fin_tfVec1.loc[model_1_not0.index]
                model_1_equal0_df = fin_tfVec1.loc[model_1_equal0.index]            
                pred = model_2.predict(model_1_not0_df)
                pred_df = pd.DataFrame(data=pred,columns=mark_df_name)   
                ###
                label1 = []
                sent1_loc_1 = []
                fin_tfVec1_loc = pd.DataFrame(columns=feature_col_name)    
                for i in range(len(labels)):
                    label1.append(pred_df[pred_df[mark_df_name[keys_index[i]]] >0 ].index)
                for j in range(len(labels)):
                    temp = fin_tfVec1.loc[label1[j]]
                    if temp.empty:     
                        pass
                    else:
                        fin_tfVec1_loc = pd.concat([fin_tfVec1_loc,temp]) 
                temp = list(fin_tfVec1_loc.index)
                for i in range(len(fin_tfVec1_loc.index)):
                    sent1_loc_1.append(df_new1.loc[temp[i]])
                sent1_loc_1 = pd.DataFrame(data=sent1_loc_1)
                sent1_loc_concat = pd.concat([sent1_loc_concat,sent1_loc_1]) 
                csv1 = convert_df(sent1_loc_concat)
                q = fin_tfVec1_loc.sum(axis = 0)
                r = pd.DataFrame(q)
                r.columns = ['sum']
                s = r.drop(r[r['sum'] == 0].index)
                t = r.drop(r[r['sum'] == 0].index).T
                t.columns.tolist()
                dict_list1_1 = []
                keys = t.columns.tolist() # 列A
                values = s['sum'].tolist() # 列B
                dict_list1_1 = dict(zip(keys, values))
                dict_list1_concat.update(dict_list1_1)
            ###
            if fin_tfVec_loc.empty & fin_tfVec1_loc.empty:
                dict_list2_concat={}
                pass
            else:   
                dict_list2_concat={}
                dict_list2_concat.update(dict_list_concat)
                dict_list2_concat.update(dict_list1_concat)
            ###
            st.set_option('deprecation.showPyplotGlobalUse', False)
            mask_image = np.array(Image.open(path+'/cloud.jpg'))
            ###
            if len(dict_list) == 0 :
                st.write(owners[0])
                st.write('無相關資訊')
            else:
                st.write(owners[0])
                wordcloud = WordCloud(background_color="white", font_path=path+'/msjh.ttf', max_words = 20, collocations=False, margin=2,mask=mask_image)
                wordcloud.generate_from_frequencies(dict_list_concat)
                plt.figure(figsize=(20,10), facecolor='k')
                plt.imshow(wordcloud)
                plt.axis("off")
                st.pyplot()
                st.download_button(owners[0]+'文本下載', data=csv, file_name=owners[0]+'文本.csv')
            if len(dict_list1) == 0 :
                st.write(owners[1])
                st.write('無相關資訊')      
            else:      
                st.write(owners[1])
                wordcloud1 = WordCloud(background_color="white", font_path=path+'/msjh.ttf', max_words = 20, collocations=False, margin=2,mask=mask_image)
                wordcloud1.generate_from_frequencies(dict_list1_concat)
                plt.figure(figsize=(20,10), facecolor='k')
                plt.imshow(wordcloud1)
                plt.axis("off")
                st.pyplot()
                st.download_button(owners[1]+'文本下載', data=csv1, file_name=owners[1]+'文本.csv')
            if len(dict_list)|len(dict_list1) == 0 :
                st.write(owners[0],'+',owners[1])
                st.write('無相關資訊')      
            else:    
                st.write(owners[0],'+',owners[1])
                wordcloud2 = WordCloud(background_color="white", font_path=path+'/msjh.ttf', max_words = 20, collocations=False, margin=2,mask=mask_image)
                wordcloud2.generate_from_frequencies(dict_list2_concat)
                plt.figure(figsize=(20,10), facecolor='k')
                plt.imshow(wordcloud2)
                plt.axis("off")
                st.pyplot()
            ###           
else: 
    pass
  