# k-means algorithm / Anton Duris(184197)
from tkinter import *
from tkinter import filedialog
from tkinter import messagebox
from tkinter import ttk

import numpy as np
import pandas as pd
# matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
from numpy.random import choice
from sklearn.datasets import make_blobs


class GUI:
    def __init__(self, gui):
        self.colmap = {1: 'r', 2: 'g', 3: 'b', 4: 'c', 5: 'y', 6: 'm', 7: 'k', 8: 'lime', 9: 'pink', 10: 'maroon'}
        self.gui = gui
        self.fig = Figure(figsize=(13, 5), dpi=100)
        self.ax = self.fig.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(self.fig, self.gui)
        self.ax.set_xlabel("X")
        self.ax.set_ylabel("Y")
        self.ax.grid()
        self.canvas.get_tk_widget().grid(row=1, column=0, columnspan=13, rowspan=5)
        # toolbar
        toolbarFrame = Frame(master=gui)
        toolbarFrame.grid(row=0, column=0)
        self.toolbar = NavigationToolbar2Tk(self.canvas, toolbarFrame)
        self.toolbar.update()
        self.ax.format_coord = lambda x, y: ""
        #####################################Step 1
        Label(self.gui, text='Step 1: Choose one from buttons below').grid(row=7, column=0, columnspan=2)
        self.browse_btn = Button(self.gui, text='Browse', command=self.browse)
        self.browse_btn.grid(row=8, column=0, columnspan=2)
        self.random_data_2d_btn = Button(self.gui, text='Random data 2d', command=self.random_data_2d)
        self.random_data_2d_btn.grid(row=9, column=0, columnspan=2)
        self.random_data_3d_btn = Button(self.gui, text='Random data 3d', command=self.random_data_3d)
        self.random_data_3d_btn.grid(row=10, column=0, columnspan=2)
        Label(self.gui, text='Random points: ').grid(row=11, column=0)
        self.points_entry = Entry(self.gui, bd=2, width=5)
        self.points_entry.insert(0, '100')
        self.points_entry.grid(row=11, column=1, sticky=W)
        Label(self.gui, text='Centers: ').grid(row=12, column=0)
        self.centers_entry = Entry(self.gui, bd=2, width=5)
        self.centers_entry.insert(0, '3')
        self.centers_entry.grid(row=12, column=1, sticky=W)
        #####################################Step 2
        Label(self.gui, text='Step 2: Automatic / Manual centroid coordiantes').grid(row=7, column=3, columnspan=3)
        # Clusters quantity
        Label(self.gui, text='Clusters quantity: ').grid(row=8, column=3)
        self.cluster_quantity = Entry(self.gui, bd=2, width=5)
        self.cluster_quantity.insert(0, '3')
        self.cluster_quantity.grid(row=8, column=4)
        # Buttons
        self.random_centroid_coordinates_2d = Button(self.gui, text='Random centroid coordinates 2D',
                                                     command=self.random_centroids)
        self.random_centroid_coordinates_2d.grid(row=9, column=3, columnspan=2)
        self.random_centroid_coordinates_3d = Button(self.gui, text='Random centroid coordinates 3D',
                                                     command=self.random_centroids_3d)
        self.random_centroid_coordinates_3d.grid(row=10, column=3, columnspan=2)
        self.random_medoid_btn = Button(self.gui, text='Random medoids', command=self.random_medoids)
        self.random_medoid_btn.grid(row=11, column=3, columnspan=2)
        self.show_centroids_btn = Button(self.gui, text='Show centroids', command=self.show_centroids)
        self.show_centroids_btn.grid(row=12, column=3, columnspan=2)
        # list of starting vectors
        Label(self.gui, text='Centroid coordinates:').grid(row=8, column=5)
        self.listbox = Text(self.gui, height=5, width=20)
        self.listbox.grid(row=9, column=5, rowspan=2)
        self.listbox.insert(END, '[]\n[]\n[]')
        #####################################Step 3
        Label(self.gui, text='Step 3: Start searching ').grid(row=7, column=6)
        # buttons
        self.search_btn = Button(main_window, text="Search for centroids", command=self.search_loop)
        self.search_btn.grid(row=8, column=6)
        self.save_btn = Button(self.gui, text='Save', command=self.save_data)
        self.save_btn.grid(row=9, column=6)
        self.clear_btn = Button(self.gui, text='Clear all', command=self.clear_all)
        self.clear_btn.grid(row=10, column=6)
        #####################################Extra
        Label(self.gui, text='Extra features').grid(row=7, column=7, columnspan=2)
        Label(self.gui, text='Select metric: ').grid(row=8, column=7)
        self.metricbox = ttk.Combobox(main_window, values=['Euclidean', 'Manhattan'])
        self.metricbox.grid(row=8, column=8)
        self.metricbox.current(0)
        # Buttons
        self.next_step_btn = Button(self.gui, text='Next step', command=self.next_step)
        self.next_step_btn.grid(row=9, column=7, columnspan=2)
        self.previous_step_btn = Button(self.gui, text='Previous step', command=self.previous_step)
        self.previous_step_btn.grid(row=10, column=7, columnspan=2)
        self.k_medoids_btn = Button(self.gui, text='K_medoids', command=self.k_medoids)
        self.k_medoids_btn.grid(row=11, column=6)

    def browse(self):
        filename = filedialog.askopenfilename(title="Select file",
                                              filetypes=(("csv files", "*.csv"), ("txt files", "*.txt*")))
        self.data = pd.read_csv(filename)

        if ('z' in self.data.keys()):
            self.state = '3d'
        else:
            self.state = '2d'
        self.centroids = {}

        # show data
        if self.state == '2d':
            self.fig.clf()
            self.ax = self.fig.gca()
            self.ax.grid()
            self.ax.scatter(self.data['x'], self.data['y'], alpha=0.5, edgecolor='k')
            self.ax.set_xlabel("X")
            self.ax.set_ylabel("Y")
            self.ax.format_coord = lambda x, y: ""
            self.canvas.draw()
        else:
            self.fig.clf()
            self.ax = self.fig.gca(projection='3d')
            self.ax.scatter(self.data['x'], self.data['y'], self.data['z'], alpha=0.5, edgecolor='k')
            self.ax.set_xlabel('X')
            self.ax.set_ylabel('Y')
            self.ax.set_zlabel('Z')
            self.ax.format_coord = lambda x, y: ""
            self.canvas.draw()

    def save_data(self):
        # save all data with all points
        # self.data.to_csv(r'data.txt', index = False)
        if self.state == '2d':
            with open('centroids.csv', 'w') as f:
                f.write("x,y\n")
                for key in self.centroids.keys():
                    f.write("%s,%s\n" % (self.centroids[key][0], self.centroids[key][1]))
        else:
            with open('centroids.csv', 'w') as f:
                f.write("x,y,z\n")
                for key in self.centroids.keys():
                    f.write("%s,%s,%s\n" % (self.centroids[key][0], self.centroids[key][1], self.centroids[key][2]))

    def clear_all(self):
        self.ax.clear()
        self.ax.grid()
        self.canvas.draw()
        self.listbox.delete('1.0', END)
        self.data.drop(self.data.index, inplace=True)

    def show_centroids(self):
        self.get_centroids()
        for i in self.centroids.keys():
            self.ax.scatter(*self.centroids[i], color=self.colmap[i])
        self.canvas.draw()

    def random_data_2d(self):
        samples = int(self.points_entry.get())
        centers_q = int(self.centers_entry.get())
        X, center_idx = make_blobs(samples, n_features=2, centers=centers_q, cluster_std=0.60, random_state=0)
        self.state = '2d'
        self.data = pd.DataFrame(dict(x=X[:, 0], y=X[:, 1]))
        self.centroids = {}
        # show data
        self.fig.clf()
        self.ax = self.fig.gca()
        self.ax.grid()
        self.ax.scatter(self.data['x'], self.data['y'], alpha=0.5, edgecolor='k')
        self.ax.set_xlabel("X")
        self.ax.set_ylabel("Y")
        self.ax.format_coord = lambda x, y: ""
        self.canvas.draw()

    def random_data_3d(self):
        samples = int(self.points_entry.get())
        centers_q = int(self.centers_entry.get())
        X, center_idx = make_blobs(samples, n_features=3, centers=centers_q, cluster_std=0.60, random_state=0)
        self.state = '3d'
        self.data = pd.DataFrame(dict(x=X[:, 0], y=X[:, 1], z=X[:, 2]))
        self.centroids = {}
        # show data
        self.fig.clf()
        self.ax = self.fig.gca(projection='3d')
        self.ax.scatter(self.data['x'], self.data['y'], self.data['z'], alpha=0.5, edgecolor='k')
        self.ax.set_xlabel('X')
        self.ax.set_ylabel('Y')
        self.ax.set_zlabel('Z')
        self.ax.format_coord = lambda x, y: ""
        self.canvas.draw()

    def random_centroids(self):
        self.state = '2d'
        # Making new centroid coordinates
        self.centroids = {i + 1: [round(np.random.uniform(self.data['x'].min(), self.data['x'].max()), 2),
                                  round(np.random.uniform(self.data['y'].min(), self.data['y'].max()), 2)] for i in
                          range(int(self.cluster_quantity.get()))}
        self.listbox.delete('1.0', END)
        for i in self.centroids.keys():
            self.listbox.insert(END, '[' + str(self.centroids[i][0]) + ',' + str(self.centroids[i][1]) + ']\n')
        # show random starting centroids
        self.fig.clf()
        self.ax = self.fig.gca()
        self.ax.grid()
        self.ax.scatter(self.data['x'], self.data['y'], alpha=0.5, edgecolor='k')
        for i in self.centroids.keys():
            self.ax.scatter(*self.centroids[i], marker='x', color=self.colmap[i])
        self.ax.set_xlabel("X")
        self.ax.set_ylabel("Y")
        self.ax.format_coord = lambda x, y: ""
        self.canvas.draw()

    def random_centroids_3d(self):
        self.state = '3d'
        # Making new centroid coordinates
        self.centroids = {i + 1: [round(np.random.uniform(self.data['x'].min(), self.data['x'].max()), 2),
                                  round(np.random.uniform(self.data['y'].min(), self.data['y'].max()), 2),
                                  round(np.random.uniform(self.data['z'].min(), self.data['z'].max()), 2)] for i in
                          range(int(self.cluster_quantity.get()))}
        self.listbox.delete('1.0', END)
        for i in self.centroids.keys():
            self.listbox.insert(END, '[' + str(self.centroids[i][0]) + ',' + str(self.centroids[i][1]) + ',' + str(
                self.centroids[i][2]) + ']\n')
        # show random starting centroids
        self.fig.clf()
        self.ax = self.fig.gca(projection='3d')
        self.ax.scatter(self.data['x'], self.data['y'], self.data['z'], alpha=0.5, edgecolor='k')
        for i in self.centroids.keys():
            self.ax.scatter(*self.centroids[i], marker='x', color=self.colmap[i])
        self.ax.set_xlabel('X')
        self.ax.set_ylabel('Y')
        self.ax.set_zlabel('Z')
        self.ax.format_coord = lambda x, y: ""
        self.canvas.draw()

    def get_centroids(self):
        for i in range(int(self.cluster_quantity.get())):
            k = i + 1
            test = str(self.listbox.get(str(i + 1) + '.0', str(i + 2) + '.0-1c'))
            if test.count(',') == 1:
                self.state = '2d'
                first = (test.split('[')[1]).split(',')[0]
                second = (test.split(',')[1]).split(']')[0]
                self.centroids.update({k: [float(first), float(second)]})
            else:
                self.state = '3d'
                first = (test.split('[')[1]).split(',')[0]
                second = (test.split(',')[1]).split(',')[0]
                third = (test.split(']')[0]).split(',')[2]
                self.centroids.update({k: [float(first), float(second), float(third)]})

    def metric_selection(self):
        metric = self.metricbox.get()
        if metric == 'Euclidean':
            # Euclidean distance between the points
            for i in self.centroids.keys():
                if self.state == '2d':
                    self.data['distance_from_{}'.format(i)] = (np.sqrt(
                        (self.data['x'] - self.centroids[i][0]) ** 2 + (self.data['y'] - self.centroids[i][1]) ** 2))
                else:
                    self.data['distance_from_{}'.format(i)] = (np.sqrt(
                        (self.data['x'] - self.centroids[i][0]) ** 2 + (self.data['y'] - self.centroids[i][1]) ** 2) + (
                                                                           self.data['z'] - self.centroids[i][2]) ** 2)
        else:
            # Manhattan Distance between points
            for i in self.centroids.keys():
                if self.state == '2d':
                    self.data['distance_from_{}'.format(i)] = ((abs(self.data['x'] - self.centroids[i][0])) + (
                        abs(self.data['y'] - self.centroids[i][1])))
                else:
                    self.data['distance_from_{}'.format(i)] = ((abs(self.data['x'] - self.centroids[i][0])) + (
                        abs(self.data['y'] - self.centroids[i][1])) + (abs(self.data['z'] - self.centroids[i][2])))

    ########################################################## Assignment Stage
    def assignment(self):
        self.metric_selection()
        centroid_distance_cols = ['distance_from_{}'.format(i) for i in self.centroids.keys()]
        # assigning points to the right centroid
        self.data['closest'] = self.data.loc[:, centroid_distance_cols].idxmin(axis=1)
        self.data['closest'] = self.data['closest'].map(lambda x: int(x.lstrip('distance_from_')))
        self.data['color'] = self.data['closest'].map(lambda x: self.colmap[x])
        return self.data

    ############################################################ Update Stage
    # finding the new centroid from clustered group of points
    def update(self):
        for i in self.centroids.keys():
            if self.state == '2d':
                self.centroids[i][0] = np.mean(self.data[self.data['closest'] == i]['x'])
                self.centroids[i][1] = np.mean(self.data[self.data['closest'] == i]['y'])
            else:
                self.centroids[i][0] = np.mean(self.data[self.data['closest'] == i]['x'])
                self.centroids[i][1] = np.mean(self.data[self.data['closest'] == i]['y'])
                self.centroids[i][2] = np.mean(self.data[self.data['closest'] == i]['z'])
        return self.centroids

    def show(self):
        if self.state == '2d':
            self.fig.clf()
            self.ax = self.fig.gca()
            self.ax.grid()
            self.ax.scatter(self.data['x'], self.data['y'], color=self.data['color'], alpha=0.5, edgecolor='k')
            for i in self.centroids.keys():
                self.ax.scatter(*self.centroids[i], marker='x', color=self.colmap[i])
            self.ax.set_xlabel("X")
            self.ax.set_ylabel("Y")
            self.ax.format_coord = lambda x, y: ""
            self.canvas.draw()
        else:
            self.fig.clf()
            self.ax = self.fig.gca(projection='3d')
            self.ax.scatter(self.data['x'], self.data['y'], self.data['z'], color=self.data['color'], alpha=0.5,
                            edgecolor='k')
            for i in self.centroids.keys():
                self.ax.scatter(*self.centroids[i], marker='x', color=self.colmap[i])
            self.ax.set_xlabel('X')
            self.ax.set_ylabel('Y')
            self.ax.set_zlabel('Z')
            self.ax.format_coord = lambda x, y: ""
            self.canvas.draw()

    def next_step(self):
        if ('closest' in self.data.keys()):
            self.old_data = self.data.copy(deep=True)
            self.old_centroids = self.centroids.copy()
            closest_centroids_steps = self.data['closest'].copy(deep=True)
            self.data = self.assignment()
            self.centroids = self.update()
            if closest_centroids_steps.equals(self.data['closest']):
                messagebox.showwarning('Warning', 'No more steps')
            else:
                self.show()
        else:
            self.old_data = self.data.copy(deep=True)
            self.old_centroids = self.centroids.copy()
            self.data = self.assignment()
            self.centroids = self.update()
            self.show()

    def previous_step(self):
        if ('closest' in self.data.keys()):
            if self.state == '2d':
                self.fig.clf()
                self.ax = self.fig.gca()
                self.ax.grid()
                self.ax.scatter(self.old_data['x'], self.old_data['y'], color=self.old_data['color'], alpha=0.5,
                                edgecolor='k')
                for i in self.old_centroids.keys():
                    self.ax.scatter(*self.old_centroids[i], marker='x', color=self.colmap[i])
                self.ax.set_xlabel("X")
                self.ax.set_ylabel("Y")
                self.ax.format_coord = lambda x, y: ""
                self.canvas.draw()
            else:
                self.fig.clf()
                self.ax = self.fig.gca(projection='3d')
                self.ax.scatter(self.old_data['x'], self.old_data['y'], self.old_data['z'],
                                color=self.old_data['color'], alpha=0.5, edgecolor='k')
                for i in self.old_centroids.keys():
                    self.ax.scatter(*self.old_centroids[i], marker='x', color=self.colmap[i])
                self.ax.set_xlabel('X')
                self.ax.set_ylabel('Y')
                self.ax.set_zlabel('Z')
                self.ax.format_coord = lambda x, y: ""
                self.canvas.draw()
        else:
            messagebox.showwarning('Warning', 'No more previous steps')

    def search_loop(self):
        self.get_centroids()
        self.data = self.assignment()
        # Continue until all assigned categories don't change any more
        while True:
            closest_centroids = self.data['closest'].copy(deep=True)
            # Update stage
            self.centroids = self.update()
            # Repeat Assigment Stage
            self.data = self.assignment()
            if closest_centroids.equals(self.data['closest']):
                break
        self.show()

    def random_medoids(self):
        samples = choice(len(self.data), size=2, replace=False)
        self.centroids = {i + 1: [self.data['x'][samples[i]], self.data['y'][samples[i]]] for i in range(len(samples))}
        # show random starting medoids
        self.fig.clf()
        self.ax = self.fig.gca()
        self.ax.grid()
        self.ax.scatter(self.data['x'], self.data['y'], alpha=0.5, edgecolor='k')
        for i in self.centroids.keys():
            self.ax.scatter(*self.centroids[i], marker='x', color=self.colmap[i])
        self.ax.set_xlabel("X")
        self.ax.set_ylabel("Y")
        self.ax.format_coord = lambda x, y: ""
        self.canvas.draw()

    ################################## pokus o K-medoids - nevydal cas na dokoncenie
    def new_medoids(self):
        for i in self.centroids.keys():
            self.medoids_cost.insert(i, self.data.loc[self.data['closest'] == i]['distance_from_' + str(i)].sum())
        # print(self.medoids_cost)

        for j in self.centroids.keys():
            # print(self.data.loc[self.data['closest'] == j])
            for i in self.data.loc[self.data['closest'] == j].index:
                # for j in self.centroids.keys():
                # self.centroids[j][0] = self.data['x'][i]
                x = self.data['x'][i]
                # self.centroids[j][1] = self.data['y'][i]
                # print(self.centroids)

    ################################# K-medoids - len prvotne priradenie k nahodnym medoidom
    def k_medoids(self):
        self.medoids_cost = []
        self.data = self.assignment()
        self.new_medoids()
        self.ax.scatter(self.data['x'], self.data['y'], color=self.data['color'], alpha=0.5, edgecolor='k')
        self.ax.format_coord = lambda x, y: ""
        self.canvas.draw()


if __name__ == '__main__':
    main_window = Tk()
    main_window.config(background='white')
    GUI(main_window)
    main_window.mainloop()
