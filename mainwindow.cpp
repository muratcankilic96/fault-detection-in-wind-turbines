#include "mainwindow.h"
#include "ui_mainwindow.h"
#include "tensorflowpreprocessor.h"
#include "dsp.h"
#include <QFileDialog>
#include <QMessageBox>

MainWindow::MainWindow(QWidget *parent)
    : QMainWindow(parent)
    , ui(new Ui::MainWindow)
{
    setWindowFlags(Qt::MSWindowsFixedSizeDialogHint);
    ui->setupUi(this);

    // Connectors
    connect(ui->workingButton, &QPushButton::clicked, this, [=]{MainWindow::file_select_button_clicked(ui->workingEdit);});
    connect(ui->problematicButton, &QPushButton::clicked, this, [=]{MainWindow::file_select_button_clicked(ui->problematicEdit);});
    connect(ui->notWorkingButton, &QPushButton::clicked, this, [=]{MainWindow::file_select_button_clicked(ui->notWorkingEdit);});
    connect(ui->workingTestButton, &QPushButton::clicked, this, [=]{MainWindow::file_select_button_clicked(ui->workingTestEdit);});
    connect(ui->problematicTestButton, &QPushButton::clicked, this, [=]{MainWindow::file_select_button_clicked(ui->problematicTestEdit);});
    connect(ui->notWorkingTestButton, &QPushButton::clicked, this, [=]{MainWindow::file_select_button_clicked(ui->notWorkingTestEdit);});
    connect(ui->deriveButton, &QPushButton::clicked, this, &MainWindow::derive_data_button_clicked);
}

MainWindow::~MainWindow()
{
    delete ui;
}

void MainWindow::file_select_button_clicked(QLineEdit * line) {
    QString path = QFileDialog::getOpenFileName(this, "Open a .wav file...", "/", "WAV File (*.wav)");
    if(!path.isEmpty())
        line->setText(path);
}

void error_message(QString text) {
    QMessageBox error;
    error.critical(0, "Error", text);
    error.setFixedSize(500, 200);
    return;
}

std::vector<fvec_t *> MainWindow::read_wav_gui(fvec_t * vec, QLineEdit * line, uint_t * sample_rate) {
    vec = read_wav_file(line->text().toUtf8().constData(), sample_rate);
    if(vec) std::cout << "Reading complete." << std::endl;
    else
    {
        error_message( "Unable to read the WAV file.");
        return {};
    }

    std::vector<fvec_t *> mat = slice_fvec(vec, ui->chunkEdit->text().toUInt());

    del_fvec(vec);

    if(!mat.empty()) std::cout << "Slicing complete." << std::endl;
    else
    {
        error_message( "Unable to chunk the WAV file into smaller parts.");
        return {};
    }

    return mat;
}

std::vector<fmat_t *> MainWindow::create_mfcc_gui(std::vector<fvec_t *> mat, uint_t sample_rate) {
    std::vector<fmat_t *> mfcc_mat_vec;
    for(int i = 0; i < (int)mat.size(); i++) {
        fmat_t * mfcc_mat = DspTools::create_mfcc(mat[i], sample_rate, 64, 32, 1024, 128);
        if(!mfcc_mat)
        {
            error_message("Unable to get MFCC representation.");
            return {};
        }
        mfcc_mat_vec.push_back(mfcc_mat);
    }
    std::cout << "MFCC complete." << std::endl;
    std::cout << "Created an instance of size (" << mfcc_mat_vec.size() << ", " << mfcc_mat_vec[0]->height << ", " << mfcc_mat_vec[0]->length << ")" << std::endl;
    return mfcc_mat_vec;
}

std::vector<fmat_t *> MainWindow::create_spectrogram_gui(std::vector<fvec_t *> mat) {
    std::vector<fmat_t *> spec_mat_vec;
    for(int i = 0; i < (int)mat.size(); i++) {
        fmat_t * spec_mat = DspTools::create_spectrogram(mat[i], 256);
        if(!spec_mat)
        {
            error_message("Unable to get spectrogram representation.");
            return {};
        }
        spec_mat_vec.push_back(spec_mat);
    }
    std::cout << "Spectrogram complete." << std::endl;
    std::cout << "Created an instance of size (" << spec_mat_vec.size() << ", " << spec_mat_vec[0]->height << ", " << spec_mat_vec[0]->length << ")" << std::endl;
    return spec_mat_vec;
}


std::vector<fmat_t *> MainWindow::create_mel_spectrogram_gui(std::vector<fvec_t *> mat) {
    std::vector<fmat_t *> mel_mat_vec;
    for(int i = 0; i < (int)mat.size(); i++) {
        fmat_t * mel_mat = DspTools::create_mel_spectrogram(mat[i], 256);
        if(!mel_mat)
        {
            error_message("Unable to get Mel spectrogram representation.");
            return {};
        }
        mel_mat_vec.push_back(mel_mat);
    }
    std::cout << "Mel spectrogram complete." << std::endl;
    std::cout << "Created an instance of size (" << mel_mat_vec.size() << ", " << mel_mat_vec[0]->height << ", " << mel_mat_vec[0]->length << ")" << std::endl;
    return mel_mat_vec;
}

void MainWindow::derive_data_button_clicked() {
    if(ui->workingEdit->text() == "" || ui->problematicEdit->text() == "" || ui->notWorkingEdit->text() == "") {
        error_message("Please fill in the empty fields.");
        return;
    }

    working_mat     = read_wav_gui(working_wav,     ui->workingEdit, &working_sample_rate);
    problematic_mat = read_wav_gui(problematic_wav, ui->problematicEdit, &problematic_sample_rate);
    not_working_mat = read_wav_gui(not_working_wav, ui->notWorkingEdit, &not_working_sample_rate);

    std::vector<fmat_t *> * working_ptr;
    std::vector<fmat_t *> * problematic_ptr;
    std::vector<fmat_t *> * not_working_ptr;

    if(ui->dataBox->currentIndex() == 0) {
        working_mfcc     = create_mfcc_gui(working_mat, working_sample_rate);
        problematic_mfcc = create_mfcc_gui(problematic_mat, problematic_sample_rate);
        not_working_mfcc = create_mfcc_gui(not_working_mat, not_working_sample_rate);
        working_ptr = &working_mfcc;
        problematic_ptr = &problematic_mfcc;
        not_working_ptr = &not_working_mfcc;

    } else if(ui->dataBox->currentIndex() == 1)  {
        working_spectrogram     = create_spectrogram_gui(working_mat);
        problematic_spectrogram = create_spectrogram_gui(problematic_mat);
        not_working_spectrogram = create_spectrogram_gui(not_working_mat);
        working_ptr = &working_spectrogram;
        problematic_ptr = &problematic_spectrogram;
        not_working_ptr = &not_working_spectrogram;
    } else if(ui->dataBox->currentIndex() == 2)  {
        working_mel_spectrogram     = create_mel_spectrogram_gui(working_mat);
        problematic_mel_spectrogram = create_mel_spectrogram_gui(problematic_mat);
        not_working_mel_spectrogram = create_mel_spectrogram_gui(not_working_mat);
        working_ptr = &working_mel_spectrogram;
        problematic_ptr = &problematic_mel_spectrogram;
        not_working_ptr = &not_working_mel_spectrogram;
    }

    tensor working_tensor = TensorflowPreprocessor::aubio_matrix_vector_to_tensor(*working_ptr);
    working_tensor = TensorflowPreprocessor::min_max_scaling(working_tensor, 0., 1.);
    TensorflowPreprocessor::to_json({working_tensor, working_tensor}, true);

    ui->trainButton->setEnabled(true);
}

