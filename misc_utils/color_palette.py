#####  Color Palette by Paletton.com
#####  Palette URL: http://paletton.com/#uid=72N0-0kmXptdvA7iLt4qwlgu+go
import matplotlib.pyplot as plt
# Primary color:
primary_red = {
    "shade_0": "#DF2F18",
    "shade_2": "#FF988A",
    "shade_3": "#FC6652",
    "shade_4": "#9C1806",
    "shade_1": "#480800"
}

secondary_orange = {
   "shade_0": "#DF7B18",
   "shade_1": "#FFE6CE",
   "shade_2": "#FEB266",
   "shade_3": "#974B00",
   "shade_4": "#381C00"
}

secondary_blue = {
   "shade_0": "#12738A",
   "shade_1": "#B8DAE2",
   "shade_2": "#428C9D",
   "shade_3": "#034C5E",
   "shade_4": "#001C23"
}

complement_green = {
   "shade_0": "#12A739",
   "shade_1": "#BEECCA",
   "shade_2": "#4CBE6A",
   "shade_3": "#00711E",
   "shade_4": "#002A0B"
}

colors = {'green': complement_green, 'red': primary_red, 'orange': secondary_orange, 'blue': secondary_blue}
#####  Generated by Paletton.com (c) 2002-2014

if __name__ == '__main__':
   fig, ax = plt.subplots()
   groups = [primary_red, secondary_blue, secondary_orange, complement_green]
   names = ["primary_red", "secondary_blue", "secondary_orange", "complement_green"]
   y = 0
   height = 6
   for idx, group in enumerate(groups):
      for shade in group.keys():
         rect = plt.Rectangle([0, y], 5, height, facecolor=group[shade])
         ax.add_patch(rect)
         plt.text(0, y + 1, shade + " " + group[shade] + " " + names[idx])
         y += height + 1
   plt.ylim([0, 142])
   plt.xlim([0, 6])
   plt.show()
