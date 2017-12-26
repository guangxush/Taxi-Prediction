#coding=utf-8
#DRM Link
# 0      1       2     3     4             5         6       7     8      9        10   11    
# LinkID MeshSq2 Node1 Node2 Administrator MajorType MajorID Major Length LinkType Auto Toll
# 12        13        14    15        16           17          18              19   20        
# Available LinkWidth Lanes LaneWidth MinLaneWidth MedianWidth MedianExtension Flow PeekSpeed 
# 21         22           23       24
# Regulation RegCondition RegSpeed Interpolations




class DRMLink:
    def __init__(self, line):
        record = line.split(',')
        self.linkid = int(record[0])                     
        self.meshsq2 = int(record[1])                     
        self.node1 = int(record[2])                      
        self.node2 = int(record[3])                      
        self.administrator = int(record[4])              
        self.majortype = int(record[5])              
        self.majorid = int(record[6])                     
        self.major = int(record[7])                     
        self.length = int(record[8])                    
        self.linktype = int(record[9])                  
        self.auto = int(record[10])                       
        self.toll = int(record[11])  
                            
        self.available = int(record[12])              
        self.linkwidth = int(record[13])                     
        self.lanes = int(record[14])                     
        self.lanewidth = int(record[15])                     
        self.minlanewidth = int(record[16])              
        self.medianwidth = int(record[17])                  
        self.medianextension = int(record[18])          
        self.flow = int(record[19])                      
        self.peekspeed = int(record[20])                 
        self.regulation = int(record[21])                
        self.regcondition = int(record[22])       
        self.regspeed = int(record[23])
        self.internumber = int(record[24])# the Interpolations contain the link's ends
        if self.internumber < 0:
            self.internumber = 0
            
        self.interlist = []
    
        for i in range(self.internumber):
            longi = float(record[25 + 2*i])
            lat = float(record[25 + 2*i + 1])
            longilat = (longi, lat)
            
            self.interlist.append(longilat)
            
    #Result:
    #  0 : node1 <-> node2
    #  1 : node1 -> node2
    #  -1 : node1 <- node2
    #  -2 : forbidden  
    # see CSV\DRMコード.xls  
    def getregulation(self):
        return {0:0, 1:0, 2:-2, 3:0, \
                4:1, 5:-1, 6:0, 7:0, \
                8:0}[self.regulation]
    
    # return length
    # majortype = 9    : * 1.4
    # majortype = 7    : * 1.2
    # majortype = else : * 1.0
    def getadjlength(self):
        if self.majortype == 9:
            return self.length*1.4
        elif self.majortype == 7:
            return self.length*1.2
        else:
            return self.length
        
    def getdir(self, n1, n2):
        if self.node1 == n1:
            return 1
        else:
            return -1  
        
    def getfirstnode(self,dir):
        if dir == 1:
            return self.node1
        else:
            return self.node2
        
    def getsecondnode(self,dir):
        if dir == 1:
            return self.node2
        else:
            return self.node1
        
    def attributes(self):
        outline = "%6d,%7d,%9d,%9d,%2d,%2d,%4d,%2d,%6d,%2d,%2d,%2d,%2d,%2d,%2d,%4d,%4d,%4d,%6d,%5d,%5d,%2d,%2d,%2d,%4d"% \
                    (self.linkid, self.meshsq2, self.node1, self.node2, self.administrator,  \
                     self.majortype, self.majorid, self.major, self.length, self.linktype, \
                     self.auto, self.toll, self.available, self.linkwidth, self.lanes, \
                     self.lanewidth, self.minlanewidth, self.medianwidth, self.medianextension, \
                     self.flow, self.peekspeed, self.regulation, self.regcondition, \
                     self.regspeed, self.internumber)
        return outline


