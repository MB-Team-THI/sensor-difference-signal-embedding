## Importing library ##
import numpy as np
from bresenham import bresenham
import matplotlib.pyplot as plt
from PIL import Image

from src.utils.exp_functions import inverse_exp_function


class SignalToImageConverter:
    def __init__(self, signal, plot_mode, image_size, max_vals, color_codes, exp_functions_params):
        self.plot_mode      = plot_mode
        self.signal         = signal
        self.color_codes    = color_codes
        self.color_x_axis   = 255
        self.sp_x_axis      = 8
        
        self.max_val_x      = max_vals['x']
        self.max_val_y      = max_vals['y']
        self.img_x_ax       = image_size['x']
        self.img_y_ax       = image_size['y']
        self.resolution_y   = {key: (self.img_y_ax/2) / self.max_val_y[key] for key in self.max_val_y}
        self.exp_functions_params = exp_functions_params
    

    def signal_to_pair(self):
        transform_data = {}
        for key in self.signal:
            if False:
                # Application of linear resolution
                sample_adapted = self.signal[key] * self.resolution_y[key]
            else:
                # Application of inverse exponential function 
                sample_adapted = inverse_exp_function(signal=self.signal[key], params=self.exp_functions_params[key])

            if self.plot_mode == "bresenham_plot":
                y1 = np.searchsorted(np.arange(-(self.img_y_ax/2), (self.img_y_ax/2)-1, 1), sample_adapted[:-1]) 
                y2 = np.searchsorted(np.arange(-(self.img_y_ax/2), (self.img_y_ax/2)-1, 1), sample_adapted[1:])
                
                x1 = np.arange(self.sp_x_axis, self.img_x_ax, (self.img_x_ax / self.max_val_x)) [:-1]
                x2 = np.arange(self.sp_x_axis, self.img_x_ax, (self.img_x_ax / self.max_val_x)) [1:]
                if max(y1) >= self.img_x_ax or max(y2) >= self.img_y_ax:
                    assert False

                temp_x = list(zip(x1, y1, x2, y2))
                transform_data[key] = (temp_x)

            elif self.plot_mode == "pyplotlib":
                transform_data[key] = sample_adapted

        return transform_data
    
    
    def bresenham_pair_to_image(self, convert):
        # plot_mode = "bresenham_plot"
        image = np.zeros((self.img_y_ax, self.img_x_ax, 3))
        # Vertical line as x-axis
        last_x_val              = int(convert[list(convert.keys())[0]][-1][2])
        bresenham_output        = list(bresenham(self.sp_x_axis, int(self.img_x_ax/2), last_x_val, int(self.img_x_ax/2)))
        img_x, img_y            = zip(*bresenham_output)
        image[img_y, img_x, :]  = self.color_x_axis

        # Per signal
        for key in convert:
            color_code = self.color_codes[key]
            # zeros.fill(0)  # Reset the values for each sample / viz multiple signals in one image
            for pair in convert[key]:
                x0, y0, x1, y1   = pair
                bresenham_output = list(bresenham(int(x0), int(y0), int(x1), int(y1)))
                img_x, img_y     = zip(*bresenham_output)
                if 'r' in color_code:
                    # Red channel
                    image[img_y, img_x, 0] = color_code['r']
                if 'g' in color_code:
                    # Green channel
                    image[img_y, img_x, 1] = color_code['g']
                if 'b' in color_code:
                    # Blue channel
                    image[img_y, img_x, 2] = color_code['b']

        return np.flipud(image)



    def plot_signals(self, signal1, signal2):
        # plot_mode = "pyplotlib"
        fig, ax = plt.subplots(figsize=(4, 4))
        
        ax.plot(signal1, color=(1, 0,0)) 
        ax.plot(signal2, color=(0, 1,0))
        
        ax.set_xlim([0, 40])
        ax.set_ylim([-128, 128])

        ax.set_facecolor('black')
        #ax.axhline(0, color='white', zorder=5) 
        x_axis = [0 for x in signal1]
        ax.plot(x_axis, color='white', linewidth=1) 

        # Removing the y axis and labels
        plt.tick_params(axis='y', which='both', left=False, right=False, labelleft=False)
        plt.xticks([])

        # Remove border
        for spine in plt.gca().spines.values():
            spine.set_visible(False)

        # Save to a file
        fig.savefig("tmp.png", bbox_inches = 'tight', pad_inches = 0, dpi=200)

        # Close the plot to free up memory
        plt.close(fig)  

        # Open an image file
        with Image.open("tmp.png") as img:
            img = img.resize((256, 256), Image.ANTIALIAS)
            # img.save(name + ".png")
        
        return img
