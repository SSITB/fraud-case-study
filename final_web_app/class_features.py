import pandas as pd
import numpy as np
from itertools import *


class Features:

    def __init__(self):
        self.X = pd.DataFrame(data=None, columns=['object_id'])


    def features_clean(self, df):
        
        self.X['object_id'] = df['object_id']
        self.X['body_length'] = df['body_length']
        self.X['name_length'] = df['name_length']
        self.X['sale_duration'] = df['sale_duration']
        self.X['user_age'] = df['user_age']
        self.X['fb_published'] = df['fb_published']
        self.X['has_logo'] = df['has_logo']
        self.X['org_facebook'] = df['org_facebook']
        self.X['org_twitter'] = df['org_twitter']
       
        #Extracting features from ticket_types dictionary
        self.X = self.ticket_types(df.loc[:,'ticket_types'])
        
        #Creating binary features
        selected_dummies = ['currency','delivery_method','payout_type']
        self.X = self.dummies(df.loc[:,selected_dummies])
        
        #Replacing missing values with the mean
        selected_vars_with_missing_vals = ['sale_duration','org_facebook','org_twitter',
                                           'avg_ticket_cost','tot_ticket_quant']
        self.X = self.replace_missing_val(self.X.loc[:,selected_vars_with_missing_vals])
        
        #Dropping object id from the features
        self.X=self.X.drop('object_id',axis=1)
        
        return self.X


    def replace_missing_val(self, df):
        
        for i in df:
            self.X[i] = self.X[i].fillna(self.X[i].mean())
        
        return self.X


    def ticket_types(self,tickets):
        
        #Extracting variables from ticket_types dictionary
        event_types = pd.DataFrame(list(chain.from_iterable(tickets)))
        #Calculating average ticket price
        event_types_aggr = pd.DataFrame(event_types['cost'].groupby(
                                        event_types['event_id']).mean())
        #Renaming average ticket price
        event_types_aggr=event_types_aggr.rename(
                                    {'cost':'avg_ticket_cost'}, axis='columns')
        #Calculating total ticket quantity
        event_types_aggr['tot_ticket_quant'] = event_types['quantity_total'].groupby(
                                        event_types['event_id']).sum()
        #Adding object id for merging purposes
        event_types_aggr['object_id'] = event_types_aggr.index
        
        #Merging self.X and ticket_types on object_id
        self.X=pd.merge(self.X, event_types_aggr, on='object_id',how='outer')

        return self.X


    def dummies(self, df):
           
        #currencies = ['AUD','CAD','EUR','GBP','NZD','USD']
        currencies = ['GBP','USD'] #Selected currencies after EDA
        
        for i in currencies:
            self.X[i]=df['currency'].where(df['currency']==i,0)
            self.X[i]=self.X[i].where(self.X[i]==0,1).astype(float)       
    
        #deliv_methods=[0.0, 1.0, 3.0]
        deliv_methods=[1.0] #Selected deliv method after EDA
        
        for i in deliv_methods:
            name = 'deliv_method_'+str(i)
            self.X[name]=df['delivery_method'].where(df['delivery_method']==i,0)
            self.X[name]=self.X[name].where(self.X[name]==0,1) 
        
        #payout_types = ['ACH','CHECK',np.nan]
        #payout_types_ = ['cash','check','missing']
        
        payout_types = ['ACH','CHECK'] #Selected payout type after EDA
        payout_types_ = ['cash','check']
        
        for i in range(len(payout_types)):
            name = 'payout_type_'+str(payout_types_[i])
            self.X[name]=df['payout_type'].where(df['payout_type']==payout_types[i],0)
            self.X[name]=self.X[name].where(self.X[name]==0,1).astype(float)     
    
        return self.X
