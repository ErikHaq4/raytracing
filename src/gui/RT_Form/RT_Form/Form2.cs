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
        string filename;
        

        private void button1_Click(object sender, EventArgs e)
        {
            //string filename;
            if (openFileDialog1.ShowDialog() == DialogResult.Cancel)
                return;
            string fileType = Path.GetExtension(openFileDialog1.FileName).ToLower();
            if (fileType == ".exe")
            {
                filename = openFileDialog1.FileName;
                string fileText = System.IO.File.ReadAllText(filename);
                label2.Text = filename;
                filename = filename.Replace(@"\", "*");

                string dir = Path.Combine(Directory.GetCurrentDirectory());
                string conf = pathUP(dir, 6);
                conf += "\\gui\\RT_Form\\RT_Form\\python_config.txt";


                using (StreamWriter streamWriter = new StreamWriter(conf))
                {
                    streamWriter.WriteLine(label2.Text);

                }

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

        private void label4_Click(object sender, EventArgs e)
        {

        }

        private void button2_Click(object sender, EventArgs e)
        {



        }

        private string pathUP(string sq, int z)
        {
            sq = Path.Combine(Directory.GetCurrentDirectory());
            string buf = sq;
            string buf1 = " ";
            for (int i = 0; i < z; i++)
            {

                var parent = Directory.GetParent(buf).FullName;
                buf = parent;

            }

            return buf;
        }

        private void Form2_Load(object sender, EventArgs e)
        {
            string dir = Path.Combine(Directory.GetCurrentDirectory());
            string conf = pathUP(dir, 6);
            conf += "\\gui\\RT_Form\\RT_Form\\python_config.txt";
            string py_cfg = " ";

            if (File.Exists(conf))
            {
                using (StreamReader streamReader = new StreamReader(conf))
                {
                   py_cfg = streamReader.ReadToEnd();
                }

                

                if (py_cfg != "NULL")
                {
                    label2.Text = py_cfg;
                    Close();
                   
                }

         

            }

            }
           
        }
    }

