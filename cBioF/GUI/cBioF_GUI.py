from tkinter import *
from tkinter.ttk import *
import sys
import os
import cBioF

from PIL import Image, ImageTk


def run_GUI():
    GUI = GUI_cBioF()
    GUI.mainloop()

class GUI_cBioF(Tk):
    """

    cBioF GUI

    """

    X = None
    y = None
    features = None

    def __init__(self, *args):

        super().__init__(*args)

        self.title("cBioF")

        # Change icon left from title
        path = os.path.dirname(os.path.abspath(__file__))
        self.iconbitmap(path + '\CBio_logo.ico')
        self.resizable(0, 0)

        # Add logo upper right
        image = Image.open(path + "\CBio_logo.tif")
        photo = ImageTk.PhotoImage(image)

        self.logo = Label(image=photo)
        self.logo.image = photo
        self.logo.grid(row=0, column=1, columnspan=2, sticky='E')

        # Read TAB
        self.tabControl = Notebook(self)  # Create Tab Control
        self.tabControl.grid(row=0, column=0, sticky='NW')  # Pack to make visible

        self.add_read_tab()
        self.add_explore_tab()
        self.add_analyse_tab()
        self.add_export_tab()

        # Add printing text:
        self.text = Text(self, wrap="word")
        self.text.grid(row=1, column=0, columnspan=2, sticky='W')
        self.text.tag_configure("stderr", foreground="#b22222")

        sys.stdout = TextRedirector(self.text, "stdout")
        sys.stderr = TextRedirector(self.text, "stderr")

        # # create a Scrollbar and associate it with txt
        # scrollx = Scrollbar(self, command=self.text.xview, orient=HORIZONTAL)
        # scrollx.grid(row=2, column=0, columnspan=2, sticky='ew')
        # self.text['xscrollcommand'] = scrollx.set

        # create a Scrollbar and associate it with txt
        scrolly = Scrollbar(self, command=self.text.yview)
        scrolly.grid(row=1, column=2, sticky='ns')
        self.text['yscrollcommand'] = scrolly.set

    def add_read_tab(self):
        # Add buttons etc. in read tab
        self.read_tab = Frame(self.tabControl)  # Create a tab
        self.tabControl.add(self.read_tab, text='Read in dataset')  # Add the tab

        self.entry_label = Label(self.read_tab, text="Directory and filename CSV file:")
        self.entry_label.grid(row=1, column=0, sticky='W')

        self.dataset_name = Entry(self.read_tab, width=50)
        self.dataset_name.grid(row=2, column=0, sticky='W')
        self.dataset_name.insert(END, 'FILENAME.csv')

        self.read_dataset = Button(self.read_tab, text="Read CSV file", command=self.read_input)
        self.read_dataset.grid(row=3, column=0, sticky='W')

    def add_explore_tab(self):
        # Add buttons etc. in explore tab
        self.expl_tab = Frame(self.tabControl)  # Create a tab
        self.tabControl.add(self.expl_tab, text='Explore dataset')  # Add the tab

        self.prep_expl = Checkbutton(self.expl_tab, text='Preprocessing')
        self.prep_expl.grid(row=1, column=0, sticky='W')

        self.class_expl = Checkbutton(self.expl_tab, text='Classification')
        self.class_expl.grid(row=2, column=0, sticky='W')

        self.missing_label = Label(self.expl_tab, text='Missing values:')
        self.missing_label.grid(row=3, column=0, sticky='W')

        self.missing_values = Entry(self.expl_tab)
        self.missing_values.grid(row=3, column=1, sticky='W')
        self.missing_values.insert(END, 'Unknown')

        self.explore_dataset = Button(self.expl_tab, text='Explore dataset', command=self.explore_input)
        self.explore_dataset.grid(row=4, column=0, sticky='W')

    def add_analyse_tab(self):

        self.ana_tab = Frame(self.tabControl)  # Create a tab
        self.tabControl.add(self.ana_tab, text='Analyse dataset')  # Add the tab

        #self.prep_ana = Checkbutton(self.ana_tab, text='Preprocessing')
        #self.prep_ana.grid(row=1, column=0, sticky='W')

        self.class_ana = Checkbutton(self.ana_tab, text='Classification')
        self.class_ana.grid(row=1, column=0, sticky='W')

        self.fs_ana = Checkbutton(self.ana_tab, text='Feature selection')
        self.fs_ana.grid(row=2, column=0, sticky='W')

        self.file_label = Label(self.ana_tab, text="File name for pipeline file:")
        self.file_label.grid(row=3, column=0, sticky='W')

        self.file_entry = Entry(self.ana_tab, width=50)
        self.file_entry.grid(row=4, column=0, sticky='W')
        self.file_entry.insert(END, 'No output file')

        self.analyse_dataset = Button(self.ana_tab, text='Explore dataset', command=self.analyse_input)
        self.analyse_dataset.grid(row=5, column=0, sticky='W')

    def add_export_tab(self):

        # Add buttons etc. in read tab
        self.export_tab = Frame(self.tabControl)  # Create a tab
        self.tabControl.add(self.export_tab, text='Export dataset')  # Add the tab

        self.export_label = Label(self.export_tab, text="Directory and filename CSV file:")
        self.export_label.grid(row=1, column=0, sticky='W')

        self.export_name = Entry(self.export_tab, width=50)
        self.export_name.grid(row=2, column=0, sticky='W')
        self.export_name.insert(END, 'FILENAME.csv')

        self.export_dataset = Button(self.export_tab, text="Export CSV file", command=self.export_data)
        self.export_dataset.grid(row=3, column=0, sticky='W')

    def read_input(self):
        self.print_progress(self.read_tab, "Reading in dataset...")
        input = self.dataset_name.get()
        try:
            self.X, self.y, self.features = cBioF.read_csv_dataset(input)
            self.print_progress(self.read_tab, "Read and imported the file %s" % input)
        except:
            self.print_progress(self.read_tab, "Invalid file name or does not exist: %s" % input)

    def explore_input(self):
        prep = self.prep_expl.instate(['selected'])
        classif = self.class_expl.instate(['selected'])
        missing = self.missing_values.get()

        if self.X is None:
            self.print_progress(self.expl_tab, "Read in a CSV dataset first")
        else:
            self.print_progress(self.expl_tab, "Exploring dataset...")
            self.X, self.y, self.features, self.expl = cBioF.explore_dataset(self.X, self.y, self.features,
                                                                             preprocessing=prep, classification=classif,
                                                                             missing_values=missing)

            if prep:
                self.print_progress(self.expl_tab, "Explored and preprocessed")
            else:
                self.print_progress(self.expl_tab, "Explored")

    def analyse_input(self):
        #prep = self.prep_ana.instate(['selected'])
        classif = self.class_ana.instate(['selected'])
        fs = self.fs_ana.instate(['selected'])
        file_name = self.file_entry.get()

        if self.X is None:
            self.print_progress(self.ana_tab, "Read in a CSV dataset first")
        else:
            self.print_progress(self.ana_tab, "Analysing dataset... (This can take several minutes)")
            pl = cBioF.analyse_dataset(self.X, self.y, self.features, preprocessing=prep, classification=classif,
                                       file_name=file_name, feature_selection=fs)

            if file_name != 'No output file':
                self.print_progress(self.ana_tab, "Analysed and exported")
            else:
                self.print_progress(self.ana_tab, "Analysed. Optimal pipeline: %s " % pl)

    def export_data(self):
        input = self.export_name.get()

        if self.X is None:
            self.print_progress(self.expl_tab, "Read in a CSV dataset first")
        else:
            self.print_progress(self.export_tab, 'Exporting dataset...')

            cBioF.export_csv_dataset(self.X, self.y, self.features, csv_path=input)
            self.print_progress(self.export_tab, "Exported dataset on %s" % input)


    def print_progress(self, print_location, print_statement):
        print('\n' + print_statement)
        #print_label = Label(print_location, text=print_statement)
        #print_label.grid(row=7, column=0, sticky='W')


class TextRedirector(object):
    def __init__(self, widget, tag="stdout"):
        self.widget = widget
        self.tag = tag

    def write(self, str):
        self.widget.configure(state="normal")
        self.widget.insert("end", str, (self.tag,))
        self.widget.configure(state="disabled")