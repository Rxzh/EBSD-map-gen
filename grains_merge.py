import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import random
from matplotlib.widgets import Button
from argparse import ArgumentParser
import os

"""
This script allows the user to merge grains by clicking on them. It's goal is to refine the twins grains merging algorithm with low tolerance
"""

# Global variables to manage color map and mode
global_cmap = None
global_colors = {}
click_mode = False  # Start in Zooming mode

def load_data(filepath):
    return np.load(filepath)

def get_cmap(grain_ids):
    """Generate a colormap only for new IDs and keep existing ones unchanged."""
    global global_cmap, global_colors
    unique_ids = np.unique(grain_ids)
    new_ids = set(unique_ids) - set(global_colors.keys())
    for id in new_ids:
        global_colors[id] = tuple(random.choices(range(256), k=3))
    
    if new_ids:  # Only update cmap if there are new ids
        sorted_ids = sorted(global_colors.keys())
        colors = [global_colors[id] for id in sorted_ids]
        global_cmap = ListedColormap(np.array(colors) / 255.0)
    
    return global_cmap

def plot_grain_ids(grain_ids, ax, fig, prev_extent):
    """Plot the grain IDs with a persistent color map, managing colorbar and zoom."""
    ax.clear()
    cmap = get_cmap(grain_ids)
    im = ax.imshow(grain_ids, cmap=cmap)
    
    if prev_extent:
        ax.axis(prev_extent)  # Restore previous zoom

    plt.draw()

def merge_grains(grain_ids, src, dest):
    """Merge grains by changing source ID to the minimum of source and destination IDs."""
    min_id = min(src, dest)
    grain_ids[grain_ids == src] = min_id
    grain_ids[grain_ids == dest] = min_id
    return grain_ids

def click_handler(event, ax, fig, grain_ids):
    """Handle click events on the plot, keeping zoom level and focus."""
    global click_mode
    if event.inaxes and click_mode:  # Check if click mode is active
        x, y = int(event.xdata), int(event.ydata)
        clicked_id = grain_ids[y, x]
        print(f"Clicked on Grain ID: {clicked_id}")
        if hasattr(click_handler, 'prev_clicked_id') and click_handler.prev_clicked_id is not None:
            print(f"Merging {clicked_id} with {click_handler.prev_clicked_id}")
            grain_ids = merge_grains(grain_ids, clicked_id, click_handler.prev_clicked_id)
            click_handler.prev_clicked_id = None
            prev_extent = ax.axis()  # Get current zoom level
            plot_grain_ids(grain_ids, ax, fig, prev_extent)  # Replot with previous zoom
        else:
            click_handler.prev_clicked_id = clicked_id
    else:
        click_handler.prev_clicked_id = None

def toggle_mode(button):
    """Toggle between clicking and zooming modes."""
    global click_mode
    click_mode = not click_mode
    button.label.set_text('Clicking' if click_mode else 'Zooming')
    print(f"Mode switched to {'Clicking' if click_mode else 'Zooming'}")


def save_data(event, grain_ids, savepath):
    """Save the grain IDs map to a file."""
    np.save(savepath, grain_ids)
    print("Data saved to "+savepath)

def main(filepath):
    global mode_button
    savepath = filepath.replace('.npy', '_refined.npy')
    if os.path.exists(savepath):
        print(f"Refined grain IDs already exist at {savepath}. Exiting...")
        res=input("Do you want to use this existing file? [Y/n]")
        if res.lower() == 'n':
            pass
        else:
            filepath = savepath
        
    grain_ids = load_data(filepath)

    fig, ax = plt.subplots(figsize=(10, 8))
    plot_grain_ids(grain_ids, ax, fig, None)
    
    
    ax_mode_button = plt.axes([0.81, 0.0, 0.1, 0.05])
    mode_button = Button(ax_mode_button, 'Zooming')  # This is the button for mode toggling
    mode_button.on_clicked(lambda event: toggle_mode(mode_button))  # Pass the button to toggle_mode

    ax_button_save = plt.axes([0.7, 0.0, 0.1, 0.05])
    button_save = Button(ax_button_save, 'Finished')
    button_save.on_clicked(lambda event: save_data(event, grain_ids, savepath))

    fig.canvas.mpl_connect('button_press_event', lambda event: click_handler(event, ax, fig, grain_ids))

    plt.show()





if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("filepath", help="Path to the grain IDs map file")
    args = parser.parse_args()

    main(args.filepath)
