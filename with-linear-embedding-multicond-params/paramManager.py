import numpy as np

import os  # for mkdir
from os import listdir, remove
from os.path import isfile, join
from scipy.interpolate import interp1d

import json

# --------------------------------------------------------
# 
# ---------------------------------------------------------

# A dictionary with some helper functions to enforce our format standard;
class pdict(dict) :
    '''A dictionary class with some helper functions to enforce our format standard:
    {'meta': {'filename' : foo,
               whatever_else},
     'param_name' : {'times' : [], 
                     'values' : [],
                     'units' : (categorical, frequency, amplitude, power, pitch, ...), default: None
                     'nvals' : number of potential values, default: 0 (means continuous real)
                     'minval: default: 0
                     'maxval: default: 1},
     ...
     }    
    
    '''
    def __init__(self, datafilename=None):
        '''Users should always use the datafilename.'''
        self['meta']={}
        if datafilename :
            self['meta']['filename']=datafilename
        
    def addMeta(self, prop, value) :
        '''Adds arbitrary properties to the meta data'''
        self['meta'][prop]=value
        
    def addParam(self, prop, times, values, units=None, nvals=0, minval=0, maxval=1, origUnits=None, origMinval=None, origMaxval=None) :
        '''Creates a parameter dictionary entry.'''
        self[prop]={}
        self[prop]['times']=times
        self[prop]['values']=values
        self[prop]['units']=units
        self[prop]['nvals']=nvals
        self[prop]['minval']=minval
        self[prop]['maxval']=maxval
        self[prop]['origMinval']=origMinval
        self[prop]['origUnits']=origUnits
        self[prop]['origMaxval']=origMaxval
       

# parameter files are json 
# Since only the dict gets json.dumped, we have to reconstruct when we load.
# If you are json.load'ing, pass the function as the object_hook parameter
# When this is passed as object_hook, all nested objects are processed with this function
def as_pdict(dct):
    if 'meta' in dct :   # if this is the 'root' ojbect, instantiate a new pdict class and set key vals
        foo=pdict()
        for key in dct : 
            foo[key]=dct[key]
        return foo
    else :
        return dct

def listDirectory_all(directory,fileExt='.wav',topdown=True):
    """returns a list of all files in directory and all its subdirectories
    directory can also take a single file.
    fileList: full path to file
    fnameList: basenames
    fnameList_noext: basenames with no extension"""
    fileList = []
    fnameList = []
    fnameList_noext = []
    if os.path.isdir(directory):
        for root, _, files in os.walk(directory, topdown=topdown):
            for name in files:
                if name.endswith(fileExt):
                    fileList.append(os.path.join(root, name))
                    fnameList.append(name)
                    fnameList_noext.append(os.path.splitext(name)[0])
    else:
        if directory.endswith(fileExt):
            fileList.append(directory)
            basename = os.path.basename(directory)
            fnameList.append(basename)
            fnameList_noext.append(os.path.splitext(basename)[0])       
    return fileList, fnameList, fnameList_noext

# json doesn't know how to encode numpy data, so we convert them to python lists
# pass this class to json.dumps as the cls parameter
#     json.dumps(data, cls=NumpyEncoder)
class NumpyEncoder(json.JSONEncoder):
    """ Special json encoder for numpy types """
    def default(self, obj):
        if isinstance(obj, (np.int_, np.intc, np.intp, np.int8,
            np.int16, np.int32, np.int64, np.uint8,
            np.uint16, np.uint32, np.uint64)):
            return int(obj)
        elif isinstance(obj, (np.float_, np.float16, np.float32, 
            np.float64)):
            return float(obj)
        elif isinstance(obj,(np.ndarray,)): #### This is the fix
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)
    
#------------------------------------------------------------------------------------

# This is the class to import to manage parameter files
#
class paramManager() :
    def __init__(self, datapath, parampath, fileExt='.wav') :
        '''Manages parameter files for the datafile stored in datapath. Creates parampath if it doesnt exist.
        paramManager can also take a single file as the datapath, in which case parampath is the parent folder of the param file'''
        self.datapath=datapath #these are the root folders
        self.parampath=parampath #init will create the same folder structure for under parampath as in datapath
        self.dataext = fileExt

        if os.path.isdir(datapath):
            for dirpath, dirnames, filenames in os.walk(datapath):
                structure = os.path.join(parampath, os.path.relpath(dirpath, datapath))
                if not os.path.isdir(structure):
                    os.mkdir(structure)
        else:
            self.filepath,self.basename = os.path.split(self.datapath)
            if len(self.filepath) == 0:
                self.filepath = '.' 
            self.shortname, _ = os.path.splitext(self.basename)
            
    def filenames(self, dir) :
        """return a list of filename in dir"""
        return [f for f in os.listdir(dir) if os.path.isfile(os.path.join(dir, f))]
           
    def initParamFiles(self, overwrite=False) : 
        '''Creates one parameter file in parampath for each data file in datapath.'''
        '''You can not addParameter(s) until the parameter files exist.'''
        if ((os.path.exists(self.parampath)) and not overwrite) :
            print("{} already exists and overite is False; Not initializing".format(self.parampath))
            return

        if os.path.isdir(self.datapath):
            for dirpath, dirnames, filenames in os.walk(self.datapath):
                if filenames: #ignore loop if list is empty
                    for name in filenames:
                        if name.endswith(self.dataext):
                            shortname, extension = os.path.splitext(name)
                            structure = os.path.join(self.parampath, os.path.relpath(dirpath, self.datapath))
                            foo=pdict(name)
                            with open(structure + '/' + shortname + '.params' , 'w') as file:
                                file.write(json.dumps(foo, cls=NumpyEncoder, indent=4)) # use `json.loads` to do the reverse
        else: #if a single file provided to paramManager instead of a directory
            foo=pdict(self.basename)
            with open(self.parampath + '/' + self.shortname + '.params' , 'w') as file:
                file.write(json.dumps(foo, cls=NumpyEncoder, indent=4))
                  
    def checkIntegrity(self) :
        '''Does a simple check for a 1-to-1 match between datafiles and paramfiles'''
        integrity=True

        # Is there a parameter file for every file in the datapath?
        if os.path.isdir(self.datapath):
            fulldatapath,_,dataname = listDirectory_all(self.datapath,fileExt=self.dataext)
            fullparampath,_,paramname = listDirectory_all(self.parampath,fileExt='.params')
            diff = list(set(dataname) - set(paramname))
            if diff:
                print("{} does not exist".format(diff))
                integrity=False
        else:
            integrity = os.path.exists(self.parampath + '/' + self.shortname + '.params') 

        # Is there a data file for every corresponding to the meta:filename stored in each param file?
        if os.path.isdir(self.datapath):
            for paramfile in fullparampath:
                with open(paramfile) as fh:
                    foo=json.load(fh)
                    structure = os.path.join(self.datapath, os.path.split(os.path.relpath(paramfile, self.parampath))[0])
                    if not os.path.isfile(structure + "/" + foo['meta']['filename']):
                        print("{} does not exist".format(structure + "/" + foo['meta']['filename']))
                        integrity=False
        else:
            with open(self.parampath + '/' + self.shortname + '.params') as fh:
                foo=json.load(fh)
                if not os.path.isfile(self.filepath + "/" + foo['meta']['filename']):
                        print("{} does not exist".format(self.filepath + "/" + foo['meta']['filename']))
                        integrity=False
        
        return integrity

    # get the parameter data structure from full path named file
    def getParams(self, pfname) :
        '''Get the pdict from the parameter file corresponding to the data file'''
        if os.path.isdir(self.datapath):
            structure = os.path.join(self.parampath, os.path.relpath(pfname, self.datapath))
            path, ext = os.path.splitext(structure)
        else:
            path = self.parampath + '/' + self.shortname
        with open(path + '.params') as fh:
            params = json.load(fh, object_hook=as_pdict)
        return params

    def getParamNames(self, pfname) :
        '''Return the param names i.e. dictionary keys in data file in a list'''
        paramdict = self.getParams(pfname)
        return [*paramdict]

    def getParamSize(self, pfname, prop) :
        '''Return the length of values/times for a specific param prop in file pfname'''
        paramdict = self.getParams(pfname)
        lentimes = len(paramdict[prop]['times'])
        lenvalues = len(paramdict[prop]['values'])
        return lentimes, lenvalues

    def getFullPathNames(self, dir) :
        '''Returns a list of the full path names for all the data files in datapath.
        You will need the datapath/filename.ext in order to process files with other libraries.'''
        flist,_,_ = listDirectory_all(dir,fileExt=self.dataext) 
        return flist
                    
    # add a parameter to the data sturcture and write the parameter file
    def addParam(self, pfname, prop, times, values, units=None, nvals=0, minval=0, maxval=1, origUnits=None, origMinval=None, origMaxval=None) :
        ''' Adds parameter data to the param file corresponding to a data file.
        pfname - data file
        prop - name of the parameter
        times - array of time points (in seconds) 
        values - array of values (must be equal in length to times)
        'units' : (categorical, frequency, amplitude, power, pitch, ...), default: None
        'nvals' : number of potential values, defualt: 0 (means continuous real)
        'minval: default: 0
        'maxval: default: 1
         '''
        if os.path.isdir(self.datapath):
            structure = os.path.join(self.parampath, os.path.relpath(pfname, self.datapath))
            path, ext = os.path.splitext(structure)
        else:
            path = self.parampath + '/' + self.shortname
        
        params = self.getParams(pfname)
        #enforce 2 values for each parameter
        if len(times)<2 or len(values)<2:
            raise ValueError("each parameter has to have at least 2 values corresponding to its start and end")
			
        #add to the param data structure
        params.addParam(prop, times, values, units, nvals, minval, maxval, origUnits, origMinval, origMaxval)
        #save the modified structure to file
        with open(path + '.params' , 'w') as file:
                file.write(json.dumps(params, cls=NumpyEncoder, indent=4)) # use `json.loads` to do the reverse

	# add more meta to the data sturcture and write the parameter file
    def addMeta(self, pfname, name, value) :
        ''' Adds additional meta data to the param file corresponding to a data file.
        pfname - data file
        name - Meta name
        value - Meta value 
        '''
        if os.path.isdir(self.datapath):
            structure = os.path.join(self.parampath, os.path.relpath(pfname, self.datapath))
            path, ext = os.path.splitext(structure)
        else:
            path = self.parampath + '/' + self.shortname
        
        params = self.getParams(pfname)
        params.addMeta(name, value)
        # save the modified structure to file
        with open(path + '.params' , 'w') as file:
            file.write(json.dumps(params, cls=NumpyEncoder, indent=4)) # use `json.loads` to do the reverse
    
    @staticmethod			
    def resample(paramvect,original_sr,resampling_sr,axis=1):
        '''resample the chosen parameter by linear interpolation (scipy's interp1d). 
        paramvect - vector of parameter values of shape (batch,length,features)
        original_sr: original sampling rate of paramvect
        resampling_sr: the sampling rate paramvect will be resampled to
        ''' 
        x = np.linspace(0,paramvect.shape[axis]/original_sr, paramvect.shape[axis])
        new_x = np.linspace(0,paramvect.shape[axis]/original_sr, resampling_sr*paramvect.shape[axis]//original_sr)
        #x = np.arange(0,paramvect.shape[axis]/original_sr, 1/original_sr)
        #new_x = np.arange(0,paramvect.shape[axis]/original_sr, 1/resampling_sr)
        try:
            new_y = interp1d(x,paramvect,fill_value="extrapolate",axis=axis)(new_x)
        except ValueError:
            print("Could not interpolate!",x,new_x)

        return new_x,new_y


    def resampleParam(self,params,prop,nsamples,timestart=None,timeend=None,verbose=False,overwrite=False,return_verbose=False):
        '''resample the chosen parameter by linear interpolation (scipy's interp1d). 
		Modifies the 'times' and 'values' entries but leaves others unchanged.
		Can resample the parameter for a chunk of audio by specifying timestart and timeend. Note: does not chunk the actual audio file. 
		Else the parameter will be resampled for the entire duration of the audio file.
        
		params - (loaded) json parameter file (output of getParams)
        prop - name of the parameter
        nsamples - number of samples between timestart and timeend i.e. len of new 'times' and 'values' list 
        timestart - start time corresponding to the audio timestamp in seconds
		timeend - end time corresponding to the audio timestamp in seconds
		verbose - prints out parameter before and after
        overwrite - overwrite the original parameter file with new values''' 
        
        #params = self.getParams(pfname)   #read existing parameter file into buffer
        if timestart is None:
            timestart = min(params[prop]['times'])
        if timeend is None:
            timeend = max(params[prop]['times'])
        if verbose:
            A = np.array(params[prop]['times'])
            B = np.array(params[prop]['values'])
            subtimes = A[(A>=timestart)&(A<=timeend)]
            subvalues = B[(A>=timestart)&(A<=timeend)]		
            print("--Data resampled from--")
            print("times:",subtimes)
            print("values:",subvalues)
        
        new_x = np.linspace(timestart, timeend, nsamples)
        try:
            new_y = interp1d(params[prop]['times'],params[prop]['values'],fill_value="extrapolate")(new_x)
        except ValueError:
            print(new_x,params[prop]['times'],params[prop]['values'])
        if verbose:
            print("--to--")
            print("times:",new_x)
            print("values:",new_y)
        #units = params[prop]['units']
        #nvals = params[prop]['nvals']
        #minval = params[prop]['minval']
        #maxval = params[prop]['maxval']

        if overwrite:
            params[prop]['times'] = new_x
            params[prop]['values'] = new_y
            filename_w_ext = params['meta']['filename']
            shortname, ext = os.path.splitext(filename_w_ext)		

            with open(self.parampath + '/' + shortname + '.params' , 'w') as file:
                file.write(json.dumps(params, cls=NumpyEncoder, indent=4))

        if return_verbose:
            assert verbose is True, 'set verbose option if want to return original times and values!'
            return new_x,new_y,subtimes,subvalues
        else:
            return new_x,new_y


    def resampleAllParams(self,params,nsamples,timestart=None,timeend=None,prop=None,verbose=False,overwrite=False):
        '''resample multiple parameters in parameter file using resampleParam method.
        prop contains the list of selected parameters. If None specified will default to all parameters (except meta).
        Will always ignore meta parameter.'''
        paramdict = {}
        if prop is None:
            prop = list(params.keys())
        for entry in prop:
            if entry != 'meta' and entry in params:
                if verbose:
                    print(entry)
                    _,value = self.resampleParam(params,entry,nsamples,timestart,timeend,verbose,overwrite)
                    print(' ')
                else:
                    _,value = self.resampleParam(params,entry,nsamples,timestart,timeend,verbose,overwrite)
                paramdict[str(entry)] = value
        return paramdict

            