using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Drawing;
using System.Text;
using System.Windows.Forms;
using System.IO;

namespace WinFormsApp4
{
    public partial class Form2 : Form
    {
        public Form2()
        {
            InitializeComponent();
        }

        private void button1_Click(object sender, EventArgs e)
        {
           string filename;
            if (openFileDialog1.ShowDialog() == DialogResult.Cancel)
                return;
            string fileType = Path.GetExtension(openFileDialog1.FileName).ToLower();
            if (fileType == ".exe" )
            {
                filename = openFileDialog1.FileName;
                string fileText = System.IO.File.ReadAllText(filename);
                label2.Text = filename;
                filename =  filename.Replace(@"\", "*");
                MessageBox.Show("Python успешно выбран");
               Close();
            }
            else
            {
                MessageBox.Show("Выбранный файл должен быть в формате python.exe , попробуйте еще раз.");
            }
        }

        private void label2_Click(object sender, EventArgs e)
        {

        }
     
    }
}
