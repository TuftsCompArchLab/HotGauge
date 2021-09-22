import sys
import random
from copy import deepcopy
from math import floor, ceil
import numpy
import PIL.Image

class GridHeatmap:
   """
   Stores a grid of temperature values
   """

   def __init__(self,name,temps,width,height):
      self.name = name
      self.temps = temps
      self.width = width
      self.height = height
      assert len(temps) == self.width * self.height , "Incorrect number of grid values:"\
                                                      "[{}x{}] != {}".format(self.width,
                                                      self.height, len(temps))

   def __repr__(self):
      return "{} Heatmap[{}x{}]\n".format(self.name, self.width, self.height)

   def clone(self, new_name = None):
      """
      Clones self and renames it

      :param new_name: the name for the clone, if None: it will be 'clone(self.name)'
      :type new_name: string, None
      """
      clone = deepcopy(self)
      if new_name == None:
         clone.name = "clone(" + clone.name + ")"
      else:
         clone.name = new_name
      return clone

   def resize(self, rows, cols, resample=PIL.Image.BILINEAR, rename=True):
      grid_size=(rows,cols)
      temp_grid = numpy.array(self.temps, dtype=numpy.float32).reshape(self.height, self.width)
      img = PIL.Image.fromarray(temp_grid, mode='F')
      new_img = img.resize(grid_size, resample=resample)
      new_temps = numpy.asarray(new_img).reshape(rows*cols)
      if rename:
          new_name = "{}_scaled_to_{}".format(self.name, grid_size)
      else:
          new_name = self.name
      new_grid = self.__class__(new_name, new_temps, cols, rows)
      return new_grid

   def average_within_mask(self,mask):
      return (self * mask) / mask

   # New heatmap with element-wise multiplication of self.temps .* other.temps
   def __mul__(self, other):
      new_heatmap = self.clone()
      if isinstance(other, (int, long, float)):
         other_name = str(other)
         for i in xrange(len(new_heatmap.temps)):
            new_heatmap.temps[i] *= other
      elif isinstance(other, (GridHeatmap)):
         assert len(self.temps) == len(other.temps)
         assert self.width == other.width
         assert self.height == other.height
         other_name = other.name
         for i in xrange(len(new_heatmap.temps)):
            new_heatmap.temps[i] *= other.temps[i]
      else:
         raise TypeError("Cannot multiply {} by {}".format(self.__class__, other.__class__))
      new_heatmap.name = "(" + self.name + "_times_" + other_name + ")"
      return new_heatmap

   __rmul__ = __mul__

   def __imul__(self, other):
      return self * other

   # New heatmap with element-wise addition of self.temps + other.temps
   def __add__(self, other):
      new_heatmap = self.clone()
      if isinstance(other, (int, long, float)):
         other_name = str(other)
         for i in xrange(len(new_heatmap.temps)):
            new_heatmap.temps[i] += other
      elif isinstance(other, (GridHeatmap)):
         assert len(self.temps) == len(other.temps)
         assert self.width == other.width
         assert self.height == other.height
         other_name = other.name
         for i in xrange(len(new_heatmap.temps)):
            new_heatmap.temps[i] += other.temps[i]
      else:
         raise TypeError("Cannot compute {} plus {}".format(self.__class__, other.__class__))
      new_heatmap.name = "(" + self.name + "_plus_" + other_name + ")"
      return new_heatmap

   __radd__ = __add__

   def __iadd__(self, other):
      return self + other

   # New heatmap with element-wise subtraction of self.temps - other.temps
   def __sub__(self, other):
      new_heatmap = self.clone()
      if isinstance(other, (int, long, float)):
         other_name = str(other)
         for i in xrange(len(new_heatmap.temps)):
            new_heatmap.temps[i] -= other
      elif isinstance(other, (GridHeatmap)):
         assert len(self.temps) == len(other.temps)
         assert self.width == other.width
         assert self.height == other.height
         other_name = other.name
         for i in xrange(len(new_heatmap.temps)):
            new_heatmap.temps[i] -= other.temps[i]
      else:
         raise TypeError("Cannot compute {} minus {}".format(self.__class__, other.__class__))
      new_heatmap.name = "(" + self.name + "_minus_" + other_name + ")"
      return new_heatmap

   __rsub__ = __sub__

   def __isub__(self, other):
      return self - other

   # ratio of the sum of temps
   def __div__(self, other):
      if isinstance(other, (GridHeatmap)):
          return self.sum_t() / other.sum_t()
      else:
          raise TypeError("Cannot divide {} by {}. Perhaps use multiply instead".format(self.__class__, other.__class__))

   # New heatmap with element-wise division of self.temps ./ other.temps
   def __idiv__(self, other):
      return self / other

   @classmethod
   def from_file(cls, file_name, width = 64, height = 64):
      #TODO: INFER GRID SIZES. Blank line = next row
      line_num = 0
      temps = []
      with open(file_name) as f:
         for line in f:
            vals = line.strip("\n").split()
            # this line is valid data
            if (len(vals) == 2):
               if(int(vals[0]) != line_num):
                  print("invalid sample number! Exiting")
                  sys.exit(-1)
               temp = (float(vals[1]))
               temps.append(temp)
               line_num += 1
      return cls(file_name,temps,width,height)

   # Define a rectangular mask from (x0,y0) to (x1,y1) on a floorplan of size width * height
   @classmethod
   def create_mask(cls,x0,x1,y0,y1,width=64,height=64,name=None):
      xmin, xmax = max(floor(x0),0), min(ceil(x1),width)
      ymin, ymax = max(floor(y0),0), min(ceil(y1),height)
      mask = []
      for i in range(0,height):
         for j in range(0,width):
            if (x0<=j and j<x1-1 and y0<=i and i<y1-1):
               mask.append(1)
            elif (xmin<=j and j<xmax and ymin<=i and i<ymax):
                if j==xmin:
                    overlap_w = (j+1)-x0
                elif j+1==xmax:
                    overlap_w = x1-j
                else:
                    overlap_w = 1

                if i==ymin:
                    overlap_h = (i+1)-y0
                elif i+1==ymax:
                    overlap_h = y1-i
                else:
                    overlap_h = 1
                err_str = "Negative overlap: y{} or x{}"
                assert overlap_h>0 and overlap_w>0, err_str.format(overlap_h, overlap_w)
                mask.append(overlap_w*overlap_h)
            else:
               mask.append(0)
      if name == None:
         name = "mask_%d,%d-%d,%d_%d,%d" % (x0,x1,y0,y1,width,height)
      return cls(name, mask, width, height)

   @classmethod
   def power_dict_to_grid(cls, flp, power_dict, power_density=True, **kwargs):
      grid = None
      for el in flp.elements:
         powers = power_dict.get(el.name, None)
         if powers:
            if power_density:
                power = numpy.mean(powers) / (el.area * 10**4) # Watts / (m^2 * cm^2/m^2)
            else:
                power = numpy.mean(powers)
            core_mask = flp.get_mask(el, **kwargs)
            power_mask = core_mask*power
            if grid==None:
               grid = power_mask
            else:
               grid += power_mask
      return grid


   # Create a heatmap of Gaussian noise of size width * height
   @classmethod
   def gaussian_noise(cls,sigma,width=64,height=64,name=None):
      mask = []
      for i in range(0,height):
         for j in range(0,width):
             mask.append(random.gauss(0, sigma))
      if name == None:
         name = "gaussian_noise(%f)" % (sigma)
      return cls(name, mask, width, height)

   # Create a heatmap of uniform noise of size width * height
   @classmethod
   def uniform_noise(cls,delta,width=64,height=64,name=None):
      mask = []
      for i in range(0,height):
         for j in range(0,width):
             mask.append(random.uniform(-delta, delta))
      if name == None:
         name = "uniform_noise(%d)" % (delta)
      return cls(name, mask, width, height)

   def get_heatmap_file_contents(self):
      s = ""
      for line_num, d in enumerate(self.temps):
         temp = ("%f"%d).rstrip("0").rstrip(".")
         s+="%d\t%s\n" %(line_num,temp)
      return s

   def write_heatmap_file(self, fname = None):
      if fname == None:
         fname = self.name
      line_num = 0
      with open(fname, 'w') as f:
         f.write(self.get_heatmap_file_contents())

   def normalize(self, value):
      self.temps = [ v - value  for v in self.temps]

   def normalize_to_min(self):
      self.normalize(self.min_t())

   def normalize_to_max(self):
      self.normalize(self.max_t())

   def normalize_to_avg(self):
      self.normalize(self.avg_t())

   def normalize_to_median(self):
      self.normalize(self.median_t())

   def sum_t(self):
      return sum(self.temps)

   def min_t(self):
      return min(self.temps)

   def max_t(self):
      return max(self.temps)

   def avg_t(self):
      return sum(self.temps)/(float(len(self.temps)))

   def median_t(self):
      if len(self.temps) < 1:
         return None
      lst = sorted(self.temps)
      if len(self.temps) %2 == 1:
         return lst[((len(lst)+1)/2)-1]
      else:
         return float(sum(lst[(len(lst)/2)-1:(len(lst)/2)+1]))/2.0

