import pandas as pd
import numpy as np
from datetime import datetime
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.neighbors import KDTree
from math import sin, cos, sqrt, atan2, radians

def distance(cord1, cord2):
    
    lat1 = float(cord1.split(',')[0][1:])
    lon1 = float(cord1.split(',')[1].replace(' ', '')[0:-1])
    lat2 = float(cord2.split(',')[0][1:])
    lon2 = float(cord2.split(',')[1].replace(' ', '')[0:-1])

    # approximate radius of earth in km
    R = 6373.0

    dlon = lon2 - lon1
    dlat = lat2 - lat1

    a = sin(dlat / 2)**2 + cos(lat1) * cos(lat2) * sin(dlon / 2)**2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))

    distance = R * c

    return distance


    
def Closest_Stores(df, df_targets, df_stores, stores_per_target):
    
    def distance(cord1, cord2):

        lat1 = float(cord1.split(',')[0][1:])
        lon1 = float(cord1.split(',')[1].replace(' ', '')[0:-1])
        lat2 = float(cord2.split(',')[0][1:])
        lon2 = float(cord2.split(',')[1].replace(' ', '')[0:-1])

        # approximate radius of earth in km
        R = 6373.0

        dlon = lon2 - lon1
        dlat = lat2 - lat1

        a = sin(dlat / 2)**2 + cos(lat1) * cos(lat2) * sin(dlon / 2)**2
        c = 2 * atan2(sqrt(a), sqrt(1 - a))

        distance = R * c

        return distance    
    
    
    df_temp = df_targets.copy()

    for storeid in df_stores.StoreID:
        df_temp.loc.__setitem__((slice(None), storeid), np.nan)
        

    for treat in df.StoreID:
        cord2 = df[df.StoreID == treat].Coordinates.values[0]
        df_temp[treat] = df_temp.Coordinates.apply(lambda x:distance(x, cord2))
        
    closed_stores = {}
    dist2_stores = {}
    sel_stores = []
    
    for i, store in enumerate(df_temp.StoreID):
        indexs = df_temp[df_temp.StoreID==store][df_temp.columns[-df.StoreID.shape[0]:]].transpose().sort_values(by=i).iloc[:80]
        count = 0
        A = []
        B = []
        for cls_str in indexs.index[1:80].tolist():
            if cls_str not in sel_stores:
                A.append(cls_str)
                count+=1
                B.append(indexs.loc[cls_str, i])
            if count == stores_per_target:
                break

        #A = [x for x in indexs.index[1:3].tolist() if x not in sel_stores]
        closed_stores[store] = A
        dist2_stores[store] = B
        for x in A:
            sel_stores.append(x)
            
    pd_df = {}
    strs = []
    dists = []
    indexs = []
    for store in (closed_stores.keys()):
        indexs.append(store)
        for store, distance in zip(closed_stores[store], dist2_stores[store]):
            strs.append(store)
            dists.append(distance)

        for n in range(stores_per_target):
            pd_df['Control_'+str(n)] = strs[n::stores_per_target]

        for n in range(2):
            pd_df['Ctrldist_'+str(n)] = dists[n::stores_per_target]

    df_clst = pd.DataFrame(pd_df, index=indexs)
    df_clst.index.names = ['Treatment']
            
    return df_clst
        
      
def Week_Analysis(ctrl_dates, treat_dates, df, store_treats, weeks = [52, 12, 12], Format='%y/%m/%d'):
    
    #Defining historical, control and treatment dates
    datetime_ctr = ctrl_dates
    datetime_tar = treat_dates

    # Generating dates for the analysis
    datetime_ctr = [datetime.strptime(datetime_ctr[0], Format).date(), datetime.strptime(datetime_ctr[1], Format).date()]
    datetime_tar = [datetime.strptime(datetime_tar[0], Format).date(), datetime.strptime(datetime_tar[1], Format).date()]
    
    # Defining indexs for the data related to the dates of interest
    hist_indexs = df[(df['Invoice_Date'] < np.datetime64(datetime_ctr[0]))].index
    ctrl_indexs = df[(df['Invoice_Date'] >= np.datetime64(datetime_ctr[0])) & 
                     (df['Invoice_Date'] <= np.datetime64(datetime_ctr[1]))].index
    treat_indexs = df[(df['Invoice_Date'] >= np.datetime64(datetime_tar[0])) & 
                      (df['Invoice_Date'] <= np.datetime64(datetime_tar[1]))].index
       
    # Identifying dates of interest with specific labels
    df.loc.__setitem__((hist_indexs, 'Label'), 'historical')
    df.loc.__setitem__((ctrl_indexs, 'Label'), 'control')
    df.loc.__setitem__((treat_indexs, 'Label'), 'treatment')
        
    # Removing stores with not enough data for the analysis, to avoid bias        
    df_temp = df[(~df.Label.isin(['control', 'treatment']))].groupby('StoreID').Invoice_Date.nunique().to_frame()
    df_temp.Invoice_Date = df_temp.Invoice_Date.apply(lambda x:np.floor(x/7))
    df_temp.reset_index(inplace=True)
    stores2drop = df_temp[(df_temp.Invoice_Date < weeks[0])].StoreID.tolist()

    for i, label in enumerate(['control', 'treatment']):
        df_temp = df[df.Label == label].groupby('StoreID').Invoice_Date.nunique().to_frame()
        df_temp.Invoice_Date = df_temp.Invoice_Date.apply(lambda x:np.ceil(x/7))
        df_temp.reset_index(inplace=True)
        stores2drop += df_temp[(df_temp.Invoice_Date < weeks[i+1])].StoreID.tolist()
        
    # Removing
    stores2drop = [store for store in list(set(stores2drop)) if store not in store_treats]
    df.drop(df[df.StoreID.isin(stores2drop)].index.tolist(), axis=0, inplace=True)
    df.reset_index(inplace=True)
    
    return df

def daily_aggregation(df_trans, label, case=0):
    
    if case==0:
        df = df_trans[df_trans.Label==label].groupby(['StoreID', 'Invoice_Date'])['Sales', 'Gross_Margin', 'QTY'].sum()
        df.reset_index(inplace=True)
        df['Transactions'] = df_trans[df_trans.Label==label].groupby(['StoreID',
                                                                               'Invoice_Date']).Invoice_Number.nunique().values
        df['SKUs'] = df_trans[df_trans.Label==label].groupby(['StoreID','Invoice_Date']).SKU.nunique().values
        
    elif case==1:
        df = df_trans[~df_trans.Label.isin(label)].groupby(['StoreID', 'Invoice_Date'])['Sales', 'Gross_Margin', 'QTY'].sum()
        df.reset_index(inplace=True)
        df['Transactions'] = df_trans[~df_trans.Label.isin(label)].groupby(['StoreID',
                                                                               'Invoice_Date']).Invoice_Number.nunique().values
        df['SKUs'] = df_trans[~df_trans.Label.isin(label)].groupby(['StoreID','Invoice_Date']).SKU.nunique().values
        
    
    return df

def weekly_aggregation(df_trans, label, case=0):
    

    if case==0:
        
        # Creating the feature Week to determine the invoice week accordingly with the date used
        dic_2_weeks = {}
        for i, date in enumerate(df_trans[df_trans.Label == label].Invoice_Date.dt.strftime('%Y-%m-%d').unique()):
            dic_2_weeks[date] = i

        # Function to determine the week
        def days2weeks(date, days_dict):
            return int(np.floor(days_dict[date]/7+1))

        # Creating the week feature
        df_trans.loc.__setitem__((df_trans[df_trans.Label == label].index, 'Week'), 
                             df_trans[df_trans.Label == label].Invoice_Date.dt.strftime('%Y-%m-%d').apply(
                                 lambda x:days2weeks(x, dic_2_weeks)))
    
        df = df_trans[df_trans.Label==label].groupby(['StoreID', 'Week'])['Sales', 'Gross_Margin', 'QTY'].sum()
        df.reset_index(inplace=True)
        df['Transactions'] = df_trans[df_trans.Label==label].groupby(['StoreID',
                                                                               'Week']).Invoice_Number.nunique().values
        df['SKUs'] = df_trans[df_trans.Label==label].groupby(['StoreID','Week']).SKU.nunique().values
        
    elif case==1:
        
        # Creating the feature Week to determine the invoice week accordingly with the date used
        dic_2_weeks = {}
        for i, week in enumerate(df_trans[~df_trans.Label.isin(label)].Invoice_Date.dt.strftime('%Y-%m-%d').unique()):
            dic_2_weeks[week] = i

        # Function to determine the week
        def days2weeks(date, days_dict):
            return int(np.ceil(days_dict[date]/7))

        # Creating the week feature
        df_trans.loc.__setitem__((df_trans[~df_trans.Label.isin(label)].index, 'Week'), 
                             df_trans[~df_trans.Label.isin(label)].Invoice_Date.dt.strftime('%Y-%m-%d').apply(
                                 lambda x:days2weeks(x, dic_2_weeks)))
    
        
        df = df_trans[~df_trans.Label.isin(label)].groupby(['StoreID', 'Week'])['Sales', 'Gross_Margin', 'QTY'].sum()
        df.reset_index(inplace=True)
        df['Transactions'] = df_trans[~df_trans.Label.isin(label)].groupby(['StoreID',
                                                                               'Week']).Invoice_Number.nunique().values
        df['SKUs'] = df_trans[~df_trans.Label.isin(label)].groupby(['StoreID','Week']).SKU.nunique().values
        
    
    return df


def Stores_Relations(treats, relations, ctrl_vars, Storespertreat, step=1000):
    
    
    strs2match = Storespertreat
    
    rel_stores = {}
    sel_stores = treats.StoreID.tolist()
    
    for store in treats.StoreID:

        indexs = treats[treats.StoreID==store][ctrl_vars].iloc[0].tolist()
        sim_stores = relations.loc[(indexs[0], indexs[1])].values[0]
        rel_stores[store] = [ID for ID in sim_stores if ID not in sel_stores][0:strs2match]
        if len(rel_stores[store])< strs2match:
            sim_stores = list(relations.loc[(indexs[0]-step, indexs[1])].values[0])
            rel_stores[store] = [ID for ID in sim_stores+rel_stores[store] if ID not in sel_stores][0:strs2match]    
        sel_stores += rel_stores[store]
    
    
    treatment_store = []
    control_store_1 = []
    control_store_2 = []

    for store in rel_stores.keys():

        treatment_store.append(store)

        for i, ctrl in enumerate(rel_stores[store]):
            if i == 0:
                control_store_1.append(ctrl)
            else:
                control_store_2.append(ctrl) 

    stores_relations = pd.DataFrame({'treatment_store':treatment_store,
                                     'control_store_1':control_store_1,
                                     'control_store_2':control_store_2})

    stores_relations.set_index('treatment_store', inplace=True)
    
    return stores_relations
 
def Define_targets(relations, region_units):
    
    treatment_store = []
    control_store = []

    for store in relations.keys():

        for i, ctrl in enumerate(relations[store]):
            treatment_store.append(store)
            control_store.append(ctrl) 

    df_targets = pd.DataFrame({'Treatment_store':treatment_store,
                                    'Control_store':control_store})

    df_targets.set_index('Treatment_store', inplace=True)
    df_targets.reset_index(inplace=True)
    df_targets = df_targets.merge(region_units, left_on='Treatment_store', 
                                  right_on='StoreID', suffixes=(False, False), how='left')
    df_targets.drop('StoreID', axis=1, inplace=True)
    
    return df_targets
    
def Treatment_Dataframe(df_targets, region_sales, comp_period, test_period, Format='%y/%m/%d'):

    datetime_ctr = [datetime.strptime(comp_period[0], Format).date(),
                    datetime.strptime(comp_period[1], Format).date()]
    datetime_tar = [datetime.strptime(test_period[0], Format).date(), 
                    datetime.strptime(test_period[1], Format).date()]

    stores = df_targets.Target.tolist()# + df_targets.Ctrl_1.tolist() + df_targets.Ctrl_2.tolist()
    
    df_tar = region_sales[region_sales.StoreID.isin(stores)].copy()
    df_tar.reset_index(drop=True, inplace=True)
    df_tar.loc[:,'Label'] = np.nan
    indexs1 = df_tar[(df_tar['Week_End'] >= np.datetime64(datetime_ctr[0])) & 
                     (df_tar['Week_End'] <= np.datetime64(datetime_ctr[1]))].index
    indexs2 = df_tar[(df_tar['Week_End'] >= np.datetime64(datetime_tar[0])) & 
                     (df_tar['Week_End'] <= np.datetime64(datetime_tar[1]))].index
    df_tar.loc.__setitem__((indexs1, 'Label'), 'Control')
    df_tar.loc.__setitem__((indexs2, 'Label'), 'Treatment')
    df_tar.reset_index(drop=True, inplace=True)
    
    return df_tar
    
    
def Control_Dataframe(df_ctrls, region_sales, comp_period, test_period, Format='%y/%m/%d'):
    
    stores1 = df_ctrls.Ctrl_1.tolist()
    stores2 = df_ctrls.Ctrl_2.tolist()
    
    datetime_ctr = [datetime.strptime(comp_period[0], Format).date(), 
                    datetime.strptime(comp_period[1], Format).date()]
    datetime_tar = [datetime.strptime(test_period[0], Format).date(), 
                    datetime.strptime(test_period[1], Format).date()]
    
    df_ctr = region_sales[region_sales.StoreID.isin(stores1)].copy()
    df_ctr.reset_index(drop=True, inplace=True)
    db = region_sales[region_sales.StoreID.isin(stores2)][['Week','Gross_Margin']].copy()
    db.reset_index(drop=True, inplace=True)
    df_ctr.loc[:,'Label'] = np.nan
    df_ctr.loc[:,'Gross_Margin_mean'] = np.nan
    df_ctr.loc[:,'Gross_Margin_2'] = np.nan
    df_ctr.loc.__setitem__((slice(None), 'Gross_Margin_2'), db.Gross_Margin)
   
    indexs1 = df_ctr[(df_ctr['Week_End'] >= np.datetime64(datetime_ctr[0])) & 
                     (df_ctr['Week_End'] <= np.datetime64(datetime_ctr[1]))].index
    indexs2 = df_ctr[(df_ctr['Week_End'] >= np.datetime64(datetime_tar[0])) & 
                     (df_ctr['Week_End'] <= np.datetime64(datetime_tar[1]))].index
    
    df_ctr.loc.__setitem__((indexs1, 'Label'), 'Control')
    df_ctr.loc.__setitem__((indexs2, 'Label'), 'Treatment')
    df_ctr.loc.__setitem__((slice(None), 'Gross_Margin_mean'), 
                           df_ctr[['Gross_Margin', 'Gross_Margin_2']].mean(axis=1))
    
    return df_ctr
    
    
def Units_distributions(Test_period, Comparison_period):
    
    sns.displot(Comparison_period[Comparison_period.Label.isin(['Control', 'Treatment'])], 
                x='Gross_Margin_mean', hue='Label', element="step")
    plt.title('Comparison_period')
    sns.displot(Test_period[Test_period.Label.isin(['Control', 'Treatment'])], 
                x='Gross_Margin', hue='Label', element="step")
    plt.title('Test_period')
    
    
def GrossMargin_plot(Test_period, Comparison_period):
    
    f, ax = plt.subplots(1, 2,figsize=(13,4))
    sns.lineplot(x='Week_End', y='Gross_Margin', data=Test_period[Test_period.Label == 'Control'], 
                 label='Treatment', ax=ax[0])
    sns.lineplot(x='Week_End', y='Gross_Margin_mean', data=Comparison_period[Comparison_period.Label == 'Control'], 
                 label='Control', ax=ax[0])
    ax[0].tick_params(axis='x', rotation=45)
    ax[0].set_title('Comparison_period')
    ax[0].grid(True)

    sns.lineplot(x='Week_End', y='Gross_Margin', data=Test_period[Test_period.Label == 'Treatment'], 
                 label='Treatment', ax=ax[1])
    sns.lineplot(x='Week_End', y='Gross_Margin_mean', data=Comparison_period[Comparison_period.Label == 'Treatment'], 
                 label='Control', ax=ax[1])
    ax[1].tick_params(axis='x', rotation=45)
    ax[1].set_title('Test_period')
    ax[1].grid(True)
    

def Boxplots(Test_period, Comparison_period):
    
    cols1 = ['Week_End', 'Gross_Margin']
    cols2 = ['Week_End', 'Gross_Margin_mean']
    dfA = Test_period[Test_period.Label=='Treatment'][cols1]
    dfA.loc.__setitem__((slice(None), 'Gross_Margin_mean'), Comparison_period[Comparison_period.Label=='Treatment'][cols2[1]])
    dfA.reset_index(drop=True, inplace=True)
    dfA.rename(columns = {'Gross_Margin_mean':'Control', 'Gross_Margin':'Treatment'}, inplace=True)
    dfB = Test_period[Test_period.Label=='Control'][cols1]
    dfB.loc.__setitem__((slice(None), 'Gross_Margin_mean'), Comparison_period[Comparison_period.Label=='Control'][cols2[1]])
    dfB.reset_index(drop=True, inplace=True)
    dfB.rename(columns = {'Gross_Margin_mean':'Control', 'Gross_Margin':'Treatment'}, inplace=True)

    f, ax = plt.subplots(1, 2,figsize=(13,4))
    sns.boxplot(data=dfB[['Control', 'Treatment']],showmeans=True, palette="coolwarm", ax=ax[0],
                meanprops={"marker":"s","markerfacecolor":"white", "markeredgecolor":"blue"})

    ax[0].set_title('Comparison_period')
    ax[0].grid(True)

    sns.lineplot(x='Week_End', y='Gross_Margin', data=Test_period[Test_period.Label == 'Treatment'], ax=ax[1])
    sns.boxplot(data=dfA[['Control', 'Treatment']],showmeans=True, palette="coolwarm", ax=ax[1],
               meanprops={"marker":"s","markerfacecolor":"white", "markeredgecolor":"blue"})
    ax[1].set_title('Test_period')
    ax[1].grid(True)
    
    
def Significance_Test(Test_period, Comparison_period, col1, col2, case=0):
    
    cols1 = ['Week_End', 'Gross_Margin']
    cols2 = ['Week_End', 'Gross_Margin_mean']
    dfA = Test_period[Test_period.Label=='Treatment'][cols1]
    dfA.loc.__setitem__((slice(None), 'Gross_Margin_mean'), Comparison_period[Comparison_period.Label=='Treatment'][cols2[1]])
    dfA.reset_index(drop=True, inplace=True)
    dfA.rename(columns = {'Gross_Margin_mean':'Control', 'Gross_Margin':'Treatment'}, inplace=True)
    dfB = Test_period[Test_period.Label=='Control'][cols1]
    dfB.loc.__setitem__((slice(None), 'Gross_Margin_mean'), Comparison_period[Comparison_period.Label=='Control'][cols2[1]])
    dfB.reset_index(drop=True, inplace=True)
    dfB.rename(columns = {'Gross_Margin_mean':'Control', 'Gross_Margin':'Treatment'}, inplace=True)    
    
    if case == 0:
        t_stat, pval = stats.ttest_rel(dfA[col1], dfB[col2])
    
    elif case == 1:
        t_stat, pval = stats.ttest_rel(dfA[col1], dfA[col2])
    
    elif case == 2:
        t_stat, pval = stats.ttest_rel(dfB[col1], dfB[col2])
    
    
    lift_up = round((dfA[col1].mean())/dfB[col1].mean(), 3)*100
    lift_down = round((dfA[col2].mean())/dfB[col2].mean(), 3)*100
    lift = abs(lift_up - lift_down)
    lifts = [lift_up, lift_down]
    
    display(pd.DataFrame({'Values':[t_stat, pval, str('{0:.2f}'.format(lift))+' %', 
                                    str('{0:.2f}'.format(-lift_up))+' %', 
                                    str('{0:.2f}'.format(lift_down))+' %']}, 
                         index=['T-stat.', 'P-val', 'Avg_Lift', 'Avg_Lift_Treatment', 'Avg_Lift_Ctrl']))
    
    return t_stat, pval, lift
    
def Boxplot2(Test_period, Comparison_period):
    
    df = pd.concat([Test_period[['StoreID', 'Week_End', 'Label', 
                                 'Gross_Margin']], Comparison_period[['StoreID', 'Week_End', 
                                                                      'Label','Gross_Margin']]])
    
    dfA = df[df.StoreID.isin(Test_period.StoreID.unique())].copy()
    dfA.loc[dfA[dfA.Label == 'Treatment'].index, 'Test_Period'] = dfA.loc[dfA[dfA.Label == 
                                                                       'Treatment'].index, 'Gross_Margin']
    dfA.loc[dfA[dfA.Label == 'Control'].index, 'Comparison_Period'] = dfA.loc[dfA[dfA.Label == 
                                                                           'Control'].index, 'Gross_Margin']

    dfB = df[df.StoreID.isin(Comparison_period.StoreID.unique())].copy()
    dfB.loc[dfB[dfB.Label == 'Treatment'].index, 'Test_Period'] = dfB.loc[dfB[dfB.Label == 'Treatment'].index, 'Gross_Margin']
    dfB.loc[dfB[dfB.Label == 'Control'].index, 'Comparison_Period'] = dfB.loc[dfB[dfB.Label == 'Control'].index, 'Gross_Margin']

    f, ax = plt.subplots(1, 2,figsize=(13,4))
    sns.boxplot(data=dfB[['Comparison_Period', 'Test_Period']],showmeans=True, palette="coolwarm", ax=ax[0],
                meanprops={"marker":"s","markerfacecolor":"white", "markeredgecolor":"blue"})

    ax[0].set_title('Control Units Analysis')
    ax[0].grid(True)

    lines = ax[0].get_lines()
    categories = ax[0].get_xticks()
    vals=[dfB[(dfB.Comparison_Period!=np.nan) & (dfB.Label=='Control')].Gross_Margin.mean(),
         dfB[(dfB.Comparison_Period!=np.nan) & (dfB.Label=='Treatment')].Gross_Margin.mean()]

    
    for cat, val in zip(categories, vals):
        # every 4th line at the interval of 6 is median line
        # 0 -> p25 1 -> p75 2 -> lower whisker 3 -> upper whisker 4 -> p50 5 -> upper extreme value
        y = round(val)

        ax[0].text(
            cat, 
            y, 
            f'{y}', 
            ha='center', 
            va='center', 
            fontweight='bold', 
            size=10,
            color='black',
            bbox=dict(facecolor='white'))
    
    
    sns.lineplot(x='Week_End', y='Gross_Margin', data=Test_period[Test_period.Label == 'Treatment'], ax=ax[1])
    sns.boxplot(data=dfA[['Comparison_Period', 'Test_Period']],showmeans=True, palette="coolwarm", ax=ax[1],
               meanprops={"marker":"s","markerfacecolor":"white", "markeredgecolor":"blue"})
    ax[1].set_title('Treatment Units Analysis')
    ax[1].grid(True)

    lines = ax[1].get_lines()
    categories = ax[1].get_xticks()
    vals=[dfA[(dfA.Comparison_Period!=np.nan) & (dfA.Label=='Control')].Gross_Margin.mean(),
         dfA[(dfA.Comparison_Period!=np.nan) & (dfA.Label=='Treatment')].Gross_Margin.mean()]

    
    for cat, val in zip(categories, vals):
        # every 4th line at the interval of 6 is median line
        # 0 -> p25 1 -> p75 2 -> lower whisker 3 -> upper whisker 4 -> p50 5 -> upper extreme value
        y = round(val)

        ax[1].text(
            cat, 
            y, 
            f'{y}', 
            ha='center', 
            va='center', 
            fontweight='bold', 
            size=10,
            color='black',
            bbox=dict(facecolor='white'))
    
    
def Corr_Control_Variables(df_stores, df_trans, ctrl_vars, treat_var):
    
    df_strs = df_stores.copy()
    df_strs.City = df_strs.City.astype('category').cat.codes
    df_strs.Region = df_strs.Region.astype('category').cat.codes
    df_strs.State = df_strs.State.astype('category').cat.codes
    
    df = df_trans.groupby(['StoreID']).sum()
    df.reset_index(inplace=True)
    df = df[['StoreID', 'Gross_Margin']].merge(df_strs, left_on='StoreID', right_on='StoreID', suffixes=(False, False))
    cols = df_strs.columns.to_list()[1:]+['Gross_Margin']
    
    display(df[cols].corr())
    
    
    plt.figure(figsize=(10, 10));
    g = sns.pairplot(df,
             y_vars=treat_var,
             x_vars=ctrl_vars,
             kind='reg',
             plot_kws={'line_kws':{'color':'red'}, 'scatter_kws': {'alpha': 0.2}});
    g.fig.suptitle("Control varibales vs. Performance metric")
    
    
def plot_trend(df_trans, df_treats):
    
    datetime_ctr = ['15/04/29', '15/07/21']
    datetime_tar = ['16/04/29', '16/07/21']
    df_trans = Week_Analysis(datetime_ctr, datetime_tar, df_trans, df_treats.StoreID.tolist())
    df_trans.head()
    df_ctrl = weekly_aggregation(df_trans, 'control')
    df_targ = weekly_aggregation(df_trans, 'treatment')
    plt.figure(figsize=(9, 4));
    sns.lineplot(x="Week", y="Transactions", label='Control', data=df_ctrl)
    sns.lineplot(x="Week", y="Transactions", label='Treatment', data=df_targ)
    plt.xlim(1, 12)
    plt.title('Gross Margin behaviour in the comparison period')
    plt.grid()
    

def training_data(df_weeksales, df_stores, comp_period, test_period, Format='%y/%m/%d'):

    # Defining dates for the clustering
    df_weekly = df_weeksales.copy()

    datetime_ctr = [datetime.strptime(comp_period[0], Format).date(),
                    datetime.strptime(comp_period[1], Format).date()]
    datetime_tar = [datetime.strptime(test_period[0], Format).date(),
                    datetime.strptime(test_period[1], Format).date()]

    # Creating the training dataset
    stores = df_stores.StoreID.unique().tolist()
    df = df_weekly[(df_weekly.StoreID.isin(stores))]
    df_train = df_weekly[(df_weekly['Week_End'] >= np.datetime64(datetime_ctr[0])) & 
                         (df_weekly['Week_End'] <= np.datetime64(datetime_ctr[1]))].copy()
    df_train.reset_index(drop=True, inplace=True)
    
    dict_vals = {}
    for i, val in enumerate(df_train.Week.unique()):
        dict_vals[val] = i+1
    df_train.Week = df_train.Week.apply(lambda x:dict_vals[x])
    
    # Creating the test dataset
    df_test = df_weekly[(df_weekly.StoreID.isin(stores)) & 
                        (df_weekly['Week_End'] >= np.datetime64(datetime_tar[0])) & 
                        (df_weekly['Week_End'] <= np.datetime64(datetime_tar[1]))].copy()
    df_test.reset_index(drop=True, inplace=True)
    dict_vals = {}
    
    for i, val in enumerate(df_test.Week.unique()):
        dict_vals[val] = i+1
    df_test['Week'] = df_test.Week.apply(lambda x:dict_vals[x])

    return df_train, df_test
    
    
def dict_closest_units(indexs, distances, df_train, df_test, df_treats):
    
    sel_ctrl_units = []
    relation_units = {}
    distance_units = {}
    for n in range(indexs.shape[0]):
        ctrl_units = []
        dist_units = []

        counter = 0
        while ((len(ctrl_units) < 2)):
            if df_train.StoreID.loc[indexs[n, counter]] not in (sel_ctrl_units + 
                                                                df_treats.StoreID.tolist()):
                sel_ctrl_units.append(df_train.StoreID.loc[indexs[n, counter]])
                ctrl_units.append(df_train.StoreID.loc[indexs[n, counter]])
                dist_units.append(distances[n, counter])

            counter +=1

        relation_units[df_test.StoreID.loc[n]] = ctrl_units
        distance_units[df_test.StoreID.loc[n]] = dist_units

    return relation_units, distance_units

def Control_Units(indexs, distances, df_train, df_test, df_treats, df_stores):
    
    relation_units, distance_units = dict_closest_units(indexs, distances, 
                                                        df_train, df_test, df_treats)
    
    df = pd.merge(pd.DataFrame(relation_units).transpose(), 
         pd.DataFrame(distance_units).transpose(), left_index=True, right_index=True)
    df.reset_index(inplace=True)
    df.rename(columns={'index':'Target','0_x':'Ctrl_1', '1_x':'Ctrl_2',
                      '0_y':'Dist2Crtl_1', '1_y':'Dist2Crtl_2'}, inplace=True)
    
    df['AvgSales_Tar'] = df_treats.AvgMonthSales
    
    df = df.merge(df_stores[df_stores.StoreID.isin(df.Ctrl_1)][['StoreID',
                  'AvgMonthSales']], left_on='Ctrl_1', right_on='StoreID', 
                  suffixes=(False, False), how='left')
    df.rename(columns={'AvgMonthSales':'AvgSales_Ctr1'}, inplace=True)
    df.drop('StoreID', axis=1, inplace=True)
    
    df = df.merge(df_stores[df_stores.StoreID.isin(df.Ctrl_2)][['StoreID', 
                  'AvgMonthSales']], left_on='Ctrl_2', right_on='StoreID', 
                  suffixes=(False, False), how='left')
    df.rename(columns={'AvgMonthSales':'AvgSales_Ctr2'}, inplace=True)
    df.drop('StoreID', axis=1, inplace=True)

    return df
    
    
    
def Treat_Ctrl_Relations(indexs, distances, df_train, df_test, df_treats):
    
    relation_units, distance_units = dict_closest_units(indexs, distances, df_train, df_test, df_treats)
    
    treatment_store = []
    control_store_1 = []
    control_store_2 = []

    for store in relation_units.keys():

        treatment_store.append(store)

        for i, ctrl in enumerate(relation_units[store]):
            if i == 0:
                control_store_1.append(ctrl)
            else:
                control_store_2.append(ctrl) 

    stores_relations = pd.DataFrame({'treatment_store':treatment_store,
                                    'control_store_1':control_store_1,
                                    'control_store_2':control_store_2})

    stores_relations.set_index('treatment_store', inplace=True)
    
    return stores_relations
    
def Closest_Stores(stores, sales, treats, region, limits, dates, 
                   leafs=10, clusters=22, Format='%y/%m/%d' ):

    stores = stores[(stores.Region.isin(region)) &
                       (stores.AvgMonthSales >= limits[0]) &
                       (stores.AvgMonthSales <= limits[1])]
    stores.reset_index(drop=True, inplace=True)
    
    sales = sales[sales.StoreID.isin(stores.StoreID)]
    sales.reset_index(drop=True, inplace=True)
    
    treats = treats[treats.Region.isin(region)]
    treats.reset_index(drop=True, inplace=True)
    
    df_train, df_test = training_data(sales, stores, 
                                      dates[0], dates[1], Format=Format)
    df_train = df_train.groupby(['StoreID']).sum()
    df_train.reset_index(inplace=True)
    
    df_train = df_train.merge(stores[['StoreID', 'AvgMonthSales']], left_on='StoreID', 
                              right_on='StoreID', suffixes=(False, False))
    
    tree = KDTree(df_train[['CountUnique_Invoice_Number', 'AvgMonthSales']], leaf_size=leafs)
    
    df_test = df_train[df_train.StoreID.isin(treats.StoreID)].groupby(['StoreID']).sum()
    df_test.reset_index(inplace=True)
    
    nearest_dist, nearest_ind = tree.query(df_test[['CountUnique_Invoice_Number', 'AvgMonthSales']], k=clusters)
        
    Relations = Control_Units(nearest_ind, nearest_dist, df_train, df_test, treats, stores)
    
    return Relations
    
    
def Significance_Test2(Test_period, Comparison_period, col1, col2):
    
    df = pd.concat([Test_period[['StoreID', 'Week_End', 'Label', 
                                 'Gross_Margin']], Comparison_period[['StoreID', 'Week_End', 
                                                                      'Label','Gross_Margin']]])
    
    dfA = df[df.StoreID.isin(Test_period.StoreID.unique())].copy()
    dfA.loc[dfA[dfA.Label == 'Treatment'].index, 'Test_Period'] = dfA.loc[dfA[dfA.Label == 
                                                                       'Treatment'].index, 'Gross_Margin']
    dfA.loc[dfA[dfA.Label == 'Control'].index, 'Comparison_Period'] = dfA.loc[dfA[dfA.Label == 
                                                                           'Control'].index, 'Gross_Margin']

    dfB = df[df.StoreID.isin(Comparison_period.StoreID.unique())].copy()
    dfB.loc[dfB[dfB.Label == 'Treatment'].index, 'Test_Period'] = dfB.loc[dfB[dfB.Label == 'Treatment'].index, 'Gross_Margin']
    dfB.loc[dfB[dfB.Label == 'Control'].index, 'Comparison_Period'] = dfB.loc[dfB[dfB.Label == 'Control'].index, 'Gross_Margin']

    t_stat, pval = stats.ttest_rel(dfA[(dfA.Comparison_Period!=np.nan) & 
                                           (dfA.Label==col1)].Gross_Margin.values, 
                                       dfA[(dfA.Comparison_Period!=np.nan) & 
                                           (dfA.Label==col2)].Gross_Margin.values)

    A1 = dfA[dfA.Label==col1].Gross_Margin.mean()
    A2 = dfA[dfA.Label==col2].Gross_Margin.mean()
    B1 = dfB[dfB.Label==col1].Gross_Margin.mean()
    B2 = dfB[dfB.Label==col2].Gross_Margin.mean()
    
    
    lift_up = -round((A2 - A1)/A2, 3)*100
    lift_down = round((B2 - B1)/B2, 3)*100
    lift = abs(lift_up - lift_down)
    lifts = [lift_up, lift_down]
    
    display(pd.DataFrame({'Values':[t_stat, pval, str('{0:.2f}'.format(lift))+' %', 
                                    str('{0:.2f}'.format(abs(lift_up)))+' %', 
                                    str('{0:.2f}'.format(abs(lift_down)))+' %']}, 
                         index=['T-stat.', 'P-val', 'Avg_Lift', 'Avg_Lift_Treatment', 'Avg_Lift_Ctrl']))
    
    return t_stat, pval, lift    