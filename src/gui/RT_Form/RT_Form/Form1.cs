using System;
using System.Collections.Generic;
using System.Drawing;
using System.Windows.Forms;
using System.IO;
using System.Diagnostics;
 using System.Management;
using System.Management.Automation;
using System.Threading.Tasks;
using System.Text;
using System.Threading;



namespace WinFormsApp4
{


    public partial class Form1 : Form
    {
        

        public Form2 z = new Form2();
        string prc1;


        //Кодировки цветов
        double r1 = 0.99;
        double g1 = 0;
        double b1 = 0;

        double r2 = 0;
        double g2 = 0.99;
        double b2 = 0;

        double r3 = 0;
        double g3 = 0;
        double b3 = 0.99;

        string kr1;
        string kref1;
        string nobj1;

        string kr2;
        string kref2;
        string nobj2;

        string kr3;
        string kref3;
        string nobj3;

        
        string prc = "C:\\Users\\Erik\\anaconda3\\python.exe";

        //текстура пола
        

        string floor = "textures/grasss1.dat";

        //пол
        double mirror = 0.01;

        public object PictureBox2 { get; private set; }

        public Form1()
        {
            InitializeComponent();

        }

        private void Form1_Load(object sender, EventArgs e)
        {
           

            z.ShowDialog();

            string dir = Path.Combine(Directory.GetCurrentDirectory());
            string pict = pathUP(dir, 6);
            string arg1 = pict + "\\scripts\\raytracing.py";
           


            //подсказки по наведению на элементы управления  

            ToolTip t = new ToolTip();
            t.SetToolTip(label4, "Трехмерная система координат для ввода центров фигур");

            ToolTip t1 = new ToolTip();
            t1.SetToolTip(label5, "Радиус окружности, описанной вокруг центра");

            ToolTip t2 = new ToolTip();
            t2.SetToolTip(label13, "Коэффициент отражения для каждой фигуры");

            ToolTip t3 = new ToolTip();
            t3.SetToolTip(label6, "Коэффициент прозрачности для каждой фигуры");

            ToolTip t4 = new ToolTip();
            t4.SetToolTip(label7, "Число правильных геометрических тел на ребре, целое число. nobj >= 0");

            ToolTip t5 = new ToolTip();
            t5.SetToolTip(trackBar2, "Угол обзора камеры");

            ToolTip t6 = new ToolTip();
            t6.SetToolTip(trackBar3, "Коэффициент отражения для пола");

            ToolTip t7 = new ToolTip();
            t7.SetToolTip(button4, "Предпросмотр с примененными настройками");

            ToolTip t8 = new ToolTip();
            t8.SetToolTip(button3, "Создание видеоролика с полными настройками");

            ToolTip t9 = new ToolTip();
            t9.SetToolTip(button1, "Открыть полученный видеофайл");

            ToolTip t10 = new ToolTip();
            t10.SetToolTip(numericUpDown2, "Коэффициент увеличения масштаба рендера со сглаживанием по алгоритму SSAA.");

            ToolTip t11 = new ToolTip();
            t11.SetToolTip(numericUpDown3, "Максимальная глубина рекурсии.");




            //текстура по умолчанию (картинка)


            //


          
            pictureBox2.Image = Image.FromFile(pict + "\\cuda\\textures\\grasss1.jpg");
            
            
           


            pictureBox1.Image = null;
            label11.Text = "90";
            trackBar2.Value = 90;

            System.Threading.Thread.CurrentThread.CurrentCulture = new System.Globalization.CultureInfo("en-US");

            label32.Text = "PC INFO: ";
            int coreCount = 0;
            foreach (var item in new System.Management.ManagementObjectSearcher("Select * from Win32_Processor").Get())
            {
                coreCount += int.Parse(item["NumberOfCores"].ToString());
            }
            label32.Text = label32.Text + "ядер =  " + coreCount;
            label32.Text = label32.Text + " лог процессоров = " + Environment.ProcessorCount;

        }
        private string pathUP(string sq, int  z)
        {
            sq = Path.Combine(Directory.GetCurrentDirectory());
            string buf =sq;
            string buf1 = " ";
            for (int i = 0; i < z; i++)
            {

                var parent = Directory.GetParent(buf).FullName;
                buf = parent;
                
            }

            return buf;
        }
      
        private void label1_Click(object sender, EventArgs e)
        {

        }

        private void button1_Click(object sender, EventArgs e)
        {



        }

        private void label19_Click(object sender, EventArgs e)
        {

        }

        private void textBox18_TextChanged(object sender, EventArgs e)
        {

        }

        private void label23_Click(object sender, EventArgs e)
        {

        }

        private void label31_Click(object sender, EventArgs e)
        {

        }

        private void radioButton1_CheckedChanged(object sender, EventArgs e)
        {
            string dir = Path.Combine(Directory.GetCurrentDirectory());
            string pict = pathUP(dir, 6);

            floor = "textures/grasss1.dat";
            pictureBox2.Image = Image.FromFile(pict + "\\cuda\\textures\\grasss1.jpg");
        }

        private void radioButton2_CheckedChanged(object sender, EventArgs e)

        {
            string dir = Path.Combine(Directory.GetCurrentDirectory());
            string pict = pathUP(dir, 6);

            floor = "textures/metall.dat";
            pictureBox2.Image = Image.FromFile(pict + "\\cuda\\textures\\metall.jpg");
        }
        private void radioButton3_CheckedChanged(object sender, EventArgs e)
        {
            string dir = Path.Combine(Directory.GetCurrentDirectory());
            string pict = pathUP(dir, 6);

            floor = "textures/derevo.dat";
            pictureBox2.Image = Image.FromFile(pict + "\\cuda\\textures\\derevo.jpg");
        }

        private async void button4_Click(object sender, EventArgs e) //ПРЕРЕНДЕР
        {
            string dir = Path.Combine(Directory.GetCurrentDirectory());
            string pict = pathUP(dir, 6);
            string arg1 = pict + "\\scripts\\raytracing.py";

            prc1 = z.label2.Text;
            prc1 = prc1.Replace(@"*", "\\");

            button1.Visible = false;
            button3.Enabled = true;
            pictureBox1.Image?.Dispose();

            kr1 = textBox14.Text;
            kref1 = textBox19.Text;
            nobj1 = textBox22.Text;

            kr2 = textBox15.Text;
            kref2 = textBox20.Text;
            nobj2 = textBox23.Text;

            kr3 = textBox16.Text;
            kref3 = textBox21.Text;
            nobj3 = textBox24.Text;

            string fastrend = " -render "
                +
              "-cube" //КУБ
               + " " + textBox1.Text.ToString() + " " + textBox2.Text.ToString() + " " + textBox3.Text.ToString() //координаты х у z
               + " " + textBox10.Text.ToString() // R окруж
               + " " + r1 + " " + g1 + " " + b1 //RGB
               + " " + "0.1" + " " + "0.1" + " " + "0" //KR KREF NOBJ
                +
               " -dodecahedron " //додекаэдр
               + " " + textBox4.Text.ToString() + " " + textBox5.Text.ToString() + " " + textBox6.Text.ToString() //координаты х у z
               + " " + textBox11.Text.ToString() // R окруж
               + " " + r2 + " " + g2 + " " + b2 //RGB
               + " " + "0.1" + " " + "0.1" + " " + "0" //KR KREF NOBJ
                +
               " -icosahedron " //додекаэдр
               + " " + textBox7.Text.ToString() + " " + textBox8.Text.ToString() + " " + textBox9.Text.ToString() //координаты х у z
               + " " + textBox12.Text.ToString() // R окруж
               + " " + r3 + " " + g3 + " " + b3 //RGB
               + " " + "0.1" + " " + "0.1" + " " + "0" //KR KREF NOBJ
               +
               

               " -ssaa" + " " + "1"
               + " -w" + " " + "400"
               + " -h" + " " + "400"
               + " -fov" + " " + label11.Text
               + " -nframes" + " " + "10"
               + " -rmax" + " " + "0"
               + " -floor -20 15 0    20 15 0    20 -15 0    -20 -15 0 " + floor + " 0 1 0    0.6";
            



          /* int i = 0;
            var task = RunProcessAsync(prc, arg1 + fastrend);
            

            while (currentThread.IsAlive)
            {
                progressBar1.Value = i % (progressBar1.Maximum + 1);
                Thread.Sleep(100);
                i += 10;

           

            await task;
           
          }*/

            RunProcessSync(prc1, arg1 + fastrend);
            RunProcessSync(prc1, arg1 + " -conv -int2png -all");
            trackBar1.Visible = true;
            trackBar1.Enabled = true;

        }
        private void trackBar1_Scroll(object sender, EventArgs e)
        {
            string dir = Path.Combine(Directory.GetCurrentDirectory());
            string pict = pathUP(dir, 6);
            string arg1 = pict + "\\scripts\\raytracing.py";

            if (pictureBox1.Image != null) { pictureBox1.Image.Dispose(); }
            pictureBox1.Image = Image.FromFile(pict + "\\cuda\\frames_png\\" + trackBar1.Value + ".png");

        }

        private async  void button3_Click(object sender, EventArgs e) // видео
        {
            string dir = Path.Combine(Directory.GetCurrentDirectory());
            string pict = pathUP(dir, 6);
            string arg1 = pict + "\\scripts\\raytracing.py";

            prc1 = z.label2.Text;
            prc1 = prc1.Replace(@"*", "\\");

            button3.Enabled = false;
            pictureBox1.Image?.Dispose();


            trackBar1.Value = trackBar1.Maximum;
            trackBar1.Enabled = false;

            string rend =

                 " -render "

                 +
                 "-cube" //КУБ
                 + " " + textBox1.Text.ToString() + " " + textBox2.Text.ToString() + " " + textBox3.Text.ToString() //координаты х у z
                 + " " + textBox10.Text.ToString() // R окруж
                 + " " + r1 + " " + g1 + " " + b1 //RGB
                 + " " + kr1 + " " + kref1 + " " + nobj1 //KR KREF NOBJ

                 +
                  " -dodecahedron" //додекаэдр
                  + " " + textBox4.Text.ToString() + " " + textBox5.Text.ToString() + " " + textBox6.Text.ToString() //координаты х у z
                  + " " + textBox11.Text.ToString() // R окруж
                  + " " + r2 + " " + g2 + " " + b2 //RGB
                  + " " + kr2 + " " + kref2 + " " + nobj2 //KR KREF NOBJ
                  +
                  " -icosahedron " //додекаэдр
                  + " " + textBox7.Text.ToString() + " " + textBox8.Text.ToString() + " " + textBox9.Text.ToString() //координаты х у z
                  + " " + textBox12.Text.ToString() // R окруж
                  + " " + r3 + " " + g3 + " " + b3 //RGB
                  + " " + kr3 + " " + kref3 + " " + nobj3 //KR KREF NOBJ


                  + " -w" + " " + textBox13.Text
                  + " -h" + " " + textBox17.Text
                  + " -ssaa" + " " + numericUpDown2.Value
                  + " -rmax" + " " + numericUpDown3.Value
                    + " -fov" + " " + label11.Text
                    + " -nframes" + " " + textBox18.Text
                    + " -floor -20 15 0    20 15 0    20 -15 0    -20 -15 0 " + floor + " 0 1 0 " + Convert.ToString(mirror)
                 ;


            RunProcessSync(prc1, arg1 + rend);
            RunProcessSync(prc1, arg1 + " -conv -int2png -all");
            RunProcessSync(prc1, arg1 + " -video " + pict+ "\\gui\\RT_Form\\RT_Form\\TMP\\" + textBox25.Text + ".mp4" + " 10");

            button1.Visible = true;

        }

        private void button6_Click(object sender, EventArgs e)
        {
            colorDialog1.FullOpen = true;
            colorDialog1.Color = this.BackColor;
            if (colorDialog1.ShowDialog() == DialogResult.Cancel)
                return;
            button6.BackColor = colorDialog1.Color;
            button6.Text = "";
            r1 = (colorDialog1.Color.R) / 255.0;
            g1 = (colorDialog1.Color.G) / 255.0;
            b1 = (colorDialog1.Color.B) / 255.0;

        }

        private void button7_Click(object sender, EventArgs e)
        {
            colorDialog1.FullOpen = true;
            colorDialog1.Color = this.BackColor;
            if (colorDialog1.ShowDialog() == DialogResult.Cancel)
                return;
            button7.BackColor = colorDialog1.Color;
            button7.Text = "";
            r2 = (colorDialog1.Color.R) / 255.0;
            g2 = (colorDialog1.Color.G) / 255.0;
            b2 = (colorDialog1.Color.B) / 255.0;
        }
        private void button8_Click_1(object sender, EventArgs e)
        {
            colorDialog1.FullOpen = true;
            colorDialog1.Color = this.BackColor;
            if (colorDialog1.ShowDialog() == DialogResult.Cancel)
                return;
            button8.BackColor = colorDialog1.Color;
            button8.Text = "";
            r3 = (colorDialog1.Color.R) / 255.0;
            g3 = (colorDialog1.Color.G) / 255.0;
            b3 = (colorDialog1.Color.B) / 255.0;
        }
        private void trackBar2_Scroll(object sender, EventArgs e)
        {
            label11.Text = "" + trackBar2.Value;
        }
        private void button5_Click(object sender, EventArgs e) //обзор
        {
            string dir = Path.Combine(Directory.GetCurrentDirectory());
            string pict = pathUP(dir, 6);
            string arg1 = pict + "\\scripts\\raytracing.py";



            string filename;
            if (openFileDialog1.ShowDialog() == DialogResult.Cancel)
                return;
            string fileType = Path.GetExtension(openFileDialog1.FileName).ToLower();
            if (fileType == ".jpg" || fileType == ".png")
            {
                filename = openFileDialog1.FileName;
                string fileText = System.IO.File.ReadAllText(filename);
                pictureBox2.Image = Image.FromFile(filename);
                MessageBox.Show("Текстура загружена");
                label26.Text = filename;
                RunProcessSync(prc, arg1 + "-conv -png2int " + filename + "  " + filename + ".dat");
                floor = filename + ".dat";
            }
            else
            {
                MessageBox.Show("Выбранный файл должен быть в формате .jpg или .png , попробуйте еще раз.");
            }
        }
        private void RunProcessSync(string prc, string param = "")
        {
            Process process = new Process();
            process.StartInfo.FileName = prc;
            if (param != "")
            {
                process.StartInfo.Arguments = param;
            }
            process.Start();

            process.WaitForExit();


        }

       /* private async Task RunProcessAsync(string prc, string param = "")
        {
         
            Process process = new Process();
            process.StartInfo.FileName = prc;
            if (param != "")
            {
                process.StartInfo.Arguments = param;
            }
           process.Start();


            Thread currentThread = Thread.CurrentThread;

            int i = 0;
            while (currentThread.IsAlive)
            {
                progressBar1.Value = i % (progressBar1.Maximum + 1);
                Thread.Sleep(100);
                i += 10;

            }
            await  Task.Run(() => process.WaitForExit());

        }

        */





        private void pictureBox2_Click(object sender, EventArgs e)
        {

        }

        private void radioButton3_CheckedChanged_1(object sender, EventArgs e)
        {
            string dir = Path.Combine(Directory.GetCurrentDirectory());
            string pict = pathUP(dir, 6);

            floor = "textures/derevo.dat";
            pictureBox2.Image = Image.FromFile(pict + "\\cuda\\textures\\derevo.jpg");

        }

        private void trackBar3_Scroll(object sender, EventArgs e)
        {
            mirror = trackBar3.Value / 100.00;
            label14.Text = Convert.ToString(mirror);

        }

        private void button1_Click_1(object sender, EventArgs e)
        {
            string dir = Path.Combine(Directory.GetCurrentDirectory());
            string pict = pathUP(dir, 6);
            string vlcpath = pathUP(dir,11);
            
            Process.Start(vlcpath + "\\Program Files\\VideoLAN\\VLC\\vlc.exe", pict + "\\gui\\RT_Form\\RT_Form\\TMP\\" + textBox25.Text + ".mp4");


        }

        private void textBox1_TextChanged(object sender, EventArgs e)
        {
          
        }
        private void textBox1_KeyPress(object sender, KeyPressEventArgs e)
        {
            if (Char.IsNumber(e.KeyChar) | (Char.IsPunctuation(e.KeyChar) | e.KeyChar == '\b' )) return;
            else
                e.Handled = true;
        }

        private void button2_Click(object sender, EventArgs e)
        {
            
            
            
           
        }
    }

}

