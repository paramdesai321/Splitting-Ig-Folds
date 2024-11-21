

def class():
   c=[]
  with open('../BestFitPlane/1CD8_BCEF_ver2.pdb' , 'r'):
      i=0
     for line in rf:     
      if(line[21] == 'B' or line[21] =='E'):
            c[i]=1
      else:
    
          c[i] = -1
         
        i+=1
 return c
