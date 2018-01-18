from pandas import read_csv
from pandas import datetime
import datetime as date_day
import pandas as pd
from statsmodels.tsa.arima_model import ARIMA
import numpy as np
def cartesian(arrays, out=None):
    arrays = [np.asarray(x) for x in arrays]
    dtype = arrays[0].dtype

    n = np.prod([x.size for x in arrays])
    if out is None:
        out = np.zeros([n, len(arrays)], dtype=dtype)

    m = n / arrays[0].size
    out[:,0] = np.repeat(arrays[0], m)
    if arrays[1:]:
        cartesian(arrays[1:], out=out[0:m,1:])
        for j in xrange(1, arrays[0].size):
            out[j*m:(j+1)*m,1:] = out[0:m,1:]
    return out

class Data_Handle(object):
    def __init__(self,trainfile,testfile,samplefile):
        self.trainfile = trainfile
        self.testfile = testfile
        self.samplefile = samplefile
        self.train_data = []
    
    def custom_sort(self,items,sort_index = 0):
        #todo fill if nan
        #sort the data based on date
        #and reformat
        unsorted_list = [each for each in items  ]
        unsorted_list.sort(key=lambda x: x[sort_index])
        #print(unsorted_list)
        #
        modified_data = self.day_route_time_map(unsorted_list)
        return modified_data
    
    def day_route_time_map(self,user_list) :
        ## break route and time of day from id
        #import pdb
        #pdb.set_trace()
        new_list =[]
        for each in user_list:
            year,month,dt_dt = [int(itm) for itm in each[0][:10].split('-')]
            t_date = date_day.date(year,month,dt_dt)
            t_day = t_date.strftime("%A")
            t_timeroute = str(each[-1])[-3:]
            #each.extend([t_date,t_day,t_timeroute])
            t_date = each[0][:10]
            t_date = datetime.strptime(t_date, '%Y-%m-%d')
            new_list.append(list(each)+[t_date,t_day,t_timeroute])
        
        #stkmap = pd.DataFrame(stks , columns=['Script','price','volume','dayrange'])
        return new_list
        
    def testfile_ops(self,):
        ## to process test data and predict output
        series = read_csv(self.testfile)
        dt_srtd_series = self.custom_sort(series.values)
        #print(dt_srtd_series)
        self.trainfile_ops()
        #import pdb
        #pdb.set_trace()
        final_result = []
        processed = []
        for each in dt_srtd_series:
            processed 
            present_process = [each[4] ,each[5]]
            if present_process not in processed:
                processed.append(present_process)
            else:
                continue
            try:
                proc_hist = self.train_data[ (self.train_data.t_day ==each[4]) & (self.train_data.t_timeroute ==each[5])]#stkmap.dayrange.quantile(.70) )]
                #df1 = proc_hist[['a','b']]
                #model = ARIMA(self.train_data, order=(5,1,0))
                proc_hist1 = proc_hist[['t_date','traffic']]
                #proc_hist1['t_date'] = pd.to_timedelta(proc_hist1['t_date'])
                t_date = [tmd for tmd in proc_hist1['t_date']]
                #t_date=t_date[1:]
                traffic = [float(trf) for trf in proc_hist1.traffic]
                #proc_hist2 = pd.DataFrame(proc_hist1,index=idx2)
                proc_hist2 = pd.DataFrame(zip(t_date,traffic) , columns=['t_date','traffic'] )# .set_index('t_date', drop=True)
                proc_hist2.index = proc_hist2.t_date
                del proc_hist2['t_date']
                proc_hist2.squeeze()
                #proc_hist2 = proc_hist2.traffic.diff().fillna(0)
                dtstp = self.date_steps(t_date[-1],each[5])
                output = self.arima_cal(proc_hist2.values,dtstp)
                #yhat = output[0]
                #print(output)
                #import pdb
                #pdb.set_trace()
                output= [int(op) if op else 0 for op in output ]
                final_result.extend(zip(dtstp,output))
                
            except Exception as e:
                #todo log
                print(present_process)
                print(e)
                raise(e)
        #    each[3][]
        self.sample_op(final_result)
    
    def sample_op(self,final_lst):
        final_lst.sort(key=lambda x: x[0])
        self.samplefile
        import csv
        with open(self.samplefile, 'wb') as myfile:
            wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
            wr.writerows([['ID','Vehicles']]+final_lst)


        #final_df = pd.DataFrame(final_result , columns=['ID','Vehicles'])
        #final_df.to_csv(self.samplefile)
        print(final_lst)
        
    def arima_cal(self,values,dtstp):
        ## ARMA forecast 
        #order = self.order_fit(values)
        def arim_internal(order=(0,0,0)):
            model = ARIMA(values, order=order)
            #order=(1,1,1)
            model_fit = model.fit(disp=0)
            #print(dtstp)
            a_output = model_fit.forecast(steps=len(dtstp))[0]
            return a_output
        
        ar_output = []
                          
        try:
            #dtstp = self.date_steps(t_date[-1],each[5])
            order = (1,1,1)#order
            ar_output = arim_internal(order=(1,1,1))
        except Exception as e:
            #raise(e)
            pass
        
        if not isinstance(ar_output, (np.ndarray, np.generic)):#not ar_output.any():
            try:
                ar_output = arim_internal()
            except Exception as e:
                raise(e)#pass
                         
        return ar_output #= arim_internal((1,1,1))
            
    def order_fit(self,values):
        ## todo to implemet orderfit by using predict and better p,d,q
        order_dict = dict()
        order_test = cartesian(([0,1, 2], [0,1], [0,1, 2]))
        
        for each_test in  order_test:
            try:
                ''
                #val =
                t_model = ARIMA(values, order=tuple(each_test))
                t_model_fit = t_model.fit(disp=0)
                val = t_model_fit.aic
                #print(val)
                if val not in order_dict:
                    order_dict[val] = tuple(each_test)
            except Exception as e:
                #todo log
                #print(e)
                pass
                #raise(e)
        #import pdb
        #pdb.set_trace()
        return order_dict.get(min(order_dict.keys()))
         
    def date_steps(self,pd_dt,pd_timeroute):
        #find how many date steps there to forecast
        step =7
        dt_list = []
        dtyear=pd_dt.month
        while(dtyear< 11):
            dt = date_day.date(pd_dt.year ,pd_dt.month ,pd_dt.day)+date_day.timedelta(days=step)
            dtyear = dt.month
            dtobj = str(dt.year)+str(dt.month).zfill(2)+str(dt.day).zfill(2)+str(pd_timeroute)
            dt_list.append(dtobj)
            step+=7
        return dt_list[:-1]
                              
    def trainfile_ops(self,):
        #load trianfile
        series = read_csv(self.trainfile)
        pdseries = [[each[2],each[4],each[5],each[6]] for each in self.custom_sort(series.values)]
        self.train_data = pd.DataFrame(pdseries , columns=['traffic','t_date','t_day','t_timeroute'])
        
        
if __name__ == "__main__":
    import os
    flpath = '/tmp/mck'
    fls =[os.path.join(flpath,each) for each in  ['train_aWnotuB.csv','test_BdBKkAj.csv','sample_submission.csv']]
    obj = Data_Handle(*fls)
    obj.testfile_ops()        