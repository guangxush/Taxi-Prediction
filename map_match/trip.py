import shapefile
import numpy as np
import sys
import os

def readtxt(files):
    data=np.loadtxt(files, delimiter=',')
    d=[]
    for i in data:
    		try:
    			d.append(int(i[1]))
    		except IndexError:
    			break
    return d

def filtershp(shpname,filterlink,shpfile):
    sf=shapefile.Reader(shpname)
    records=sf.records()
    length=len(records)
    print "\nThe length of the shp is: "+str(length)
    fullindex=range(length)
    for j in range(length-1,-1,-1):
        for i in filterlink:
            if records[j][0]==i:
                fullindex.pop(j)
    fullindex.sort(reverse=True)
    print '\nFiltershp finished. Waiting for editing...'
    w=shapefile.Editor(shpname)
    for i in fullindex:
        w.delete(i)
    w.save('./shp/'+shpfile)
    print '\nCompleted.'

def main():
    filelist=os.listdir('./path/15192/')
    print '\nSave the trip of taxi into shp file...'
    for filepath in filelist:
    		print filepath
    		filterlink = readtxt('path/15192/'+filepath)
    		filterindex = filtershp('Road/Road.shp', filterlink, filepath)

#layer = iface.addVectorLayer("E:\ligroup\FCD\ShanghaiGIS\Roadx.shp", "Roadx", "ogr")

if __name__ == "__main__":
    main()
