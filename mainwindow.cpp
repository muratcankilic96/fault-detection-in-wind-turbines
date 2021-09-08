#include "mainwindow.h"
#include "ui_mainwindow.h"
#include "dsp.h"
#include <QFileDialog>
#include <QMessageBox>
#include <fstream>

MainWindow::MainWindow(QWidget *parent)
    : QMainWindow(parent)
    , ui(new Ui::MainWindow)
{
    setWindowFlags(Qt::MSWindowsFixedSizeDialogHint);
    ui->setupUi(this);
    process = new QProcess(this);
    process_2 = new QProcess(this);
    // Connectors
    connect(ui->workingButton, &QPushButton::clicked, this, [=]{MainWindow::file_select_button_clicked(ui->workingEdit);});
    connect(ui->problematicButton, &QPushButton::clicked, this, [=]{MainWindow::file_select_button_clicked(ui->problematicEdit);});
    connect(ui->notWorkingButton, &QPushButton::clicked, this, [=]{MainWindow::file_select_button_clicked(ui->notWorkingEdit);});
    connect(ui->workingTestButton, &QPushButton::clicked, this, [=]{MainWindow::file_select_button_clicked(ui->workingTestEdit);});
    connect(ui->problematicTestButton, &QPushButton::clicked, this, [=]{MainWindow::file_select_button_clicked(ui->problematicTestEdit);});
    connect(ui->notWorkingTestButton, &QPushButton::clicked, this, [=]{MainWindow::file_select_button_clicked(ui->notWorkingTestEdit);});
    connect(ui->deriveButton, &QPushButton::clicked, this, &MainWindow::derive_data_button_clicked);
    connect(ui->deriveButton_2, &QPushButton::clicked, this, &MainWindow::test_derive_data_button_clicked);
    connect(ui->trainButton, &QPushButton::clicked, this, &MainWindow::train_button_clicked);
    connect(ui->loadButton, &QPushButton::clicked, this, &MainWindow::load_button_clicked);
    connect(ui->testModelAccuracyButton, &QPushButton::clicked, this, &MainWindow::test_accuracy_button_clicked);
    connect(process, &QProcess::readyReadStandardOutput, this, &MainWindow::process_stdout);
    connect(process, &QProcess::readyReadStandardError, this, &MainWindow::process_stderr);
    connect(process, &QProcess::finished, this, &MainWindow::process_end);
    connect(process_2, &QProcess::started, this, &MainWindow::process_2_start);
    connect(process_2, &QProcess::finished, this, &MainWindow::process_2_end);

    // Hide tab at start
    ui->tabsWidget->removeTab(2);
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

void MainWindow::error_message(QString text) {
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

std::vector<tensor> MainWindow::preprocess_data(QLineEdit * wav1, QLineEdit * wav2, QLineEdit * wav3, bool set_data_type) {
    working_mat     = read_wav_gui(working_wav,     wav1, &working_sample_rate);
    if(working_mat.empty())
        return {};

    problematic_mat = read_wav_gui(problematic_wav, wav2, &problematic_sample_rate);
    if(problematic_mat.empty())
        return {};

    not_working_mat = read_wav_gui(not_working_wav, wav3, &not_working_sample_rate);
    if(not_working_mat.empty())
        return {};

    std::vector<fmat_t *> * working_ptr;
    std::vector<fmat_t *> * problematic_ptr;
    std::vector<fmat_t *> * not_working_ptr;

    int selected_id = ui->modelSelectBox->currentIndex() - 1;

    int data_type;
    if(set_data_type)
        data_type  = ui->dataBox->currentIndex();
    else
        data_type = (model_id_saved[selected_id] - 1) % 3;

    if(data_type == 0) {
        working_mfcc     = create_mfcc_gui(working_mat, working_sample_rate);
        if(working_mfcc.empty())
            return {};
        problematic_mfcc = create_mfcc_gui(problematic_mat, problematic_sample_rate);
        if(problematic_mfcc.empty())
            return {};
        not_working_mfcc = create_mfcc_gui(not_working_mat, not_working_sample_rate);
        if(not_working_mfcc.empty())
            return {};
        working_ptr = &working_mfcc;
        problematic_ptr = &problematic_mfcc;
        not_working_ptr = &not_working_mfcc;

    } else if(data_type == 1)  {
        working_spectrogram     = create_spectrogram_gui(working_mat);
        if(working_spectrogram.empty())
            return {};
        problematic_spectrogram = create_spectrogram_gui(problematic_mat);
        if(problematic_spectrogram.empty())
            return {};
        not_working_spectrogram = create_spectrogram_gui(not_working_mat);
        if(not_working_spectrogram.empty())
            return {};
        working_ptr = &working_spectrogram;
        problematic_ptr = &problematic_spectrogram;
        not_working_ptr = &not_working_spectrogram;
    } else if(data_type == 2)  {
        working_mel_spectrogram     = create_mel_spectrogram_gui(working_mat);
        if(working_mel_spectrogram.empty())
            return {};
        problematic_mel_spectrogram = create_mel_spectrogram_gui(problematic_mat);
        if(problematic_mel_spectrogram.empty())
            return {};
        not_working_mel_spectrogram = create_mel_spectrogram_gui(not_working_mat);
        if(not_working_mel_spectrogram.empty())
            return {};
        working_ptr = &working_mel_spectrogram;
        problematic_ptr = &problematic_mel_spectrogram;
        not_working_ptr = &not_working_mel_spectrogram;
    }

    tensor working_tensor = TensorflowPreprocessor::aubio_matrix_vector_to_tensor(*working_ptr);
    tensor problematic_tensor = TensorflowPreprocessor::aubio_matrix_vector_to_tensor(*problematic_ptr);
    tensor not_working_tensor = TensorflowPreprocessor::aubio_matrix_vector_to_tensor(*not_working_ptr);
    return {working_tensor, problematic_tensor, not_working_tensor};
}

void MainWindow::derive_data_button_clicked() {
    if(ui->workingEdit->text() == "" || ui->problematicEdit->text() == "" || ui->notWorkingEdit->text() == "") {
        error_message("Please fill in the empty fields.");
        return;
    }

    int data_type  = ui->dataBox->currentIndex();
    int model_type = ui->modelBox->currentIndex();
    float lower_bound;

    int model_id = model_type * 3 + (data_type + 1);

    if(model_type < 2)
        lower_bound = -1.;
    else
        lower_bound = 0.;

    std::vector<tensor> tensors = preprocess_data(ui->workingEdit, ui->problematicEdit, ui->notWorkingEdit, true);
    if(tensors.empty()) {
        error_message("Due to errors at training data, the process halted.");
        return;
    }
    for(int i = 0; i < tensors.size(); i++)
        tensors[i] = TensorflowPreprocessor::min_max_scaling(tensors[i], lower_bound, 1.);
    TensorflowPreprocessor::to_json("train.json", {tensors[0], tensors[1], tensors[2]}, ui->epochEdit->text().toUInt(), ui->SMOTECheckBox->isChecked(), model_id);
    tensors.clear();

    if(ui->deriveTest->isChecked()) {
        std::vector<tensor> tensors_test = preprocess_data(ui->workingTestEdit, ui->problematicTestEdit, ui->notWorkingTestEdit, true);
        if(tensors_test.empty()) {
            error_message("Due to errors at test data, the process halted.");
            return;
        }
        for(int i = 0; i < tensors_test.size(); i++)
            tensors_test[i] = TensorflowPreprocessor::min_max_scaling(tensors_test[i], lower_bound, 1.);
        TensorflowPreprocessor::to_json("test.json", {tensors_test[0], tensors_test[1], tensors_test[2]}, ui->SMOTECheckBox->isChecked());
        tensors_test.clear();
    }

    ui->trainButton->setEnabled(true);
}


void MainWindow::test_derive_data_button_clicked() {
    if(ui->workingTestEdit->text() == "" || ui->problematicTestEdit->text() == "" || ui->notWorkingTestEdit->text() == "") {
        error_message("Please fill in the empty fields.");
        return;
    }

    float lower_bound;

    int selected_id = ui->modelSelectBox->currentIndex() - 1;

    if(model_id_saved[selected_id] <= 6)
        lower_bound = -1.;
    else
        lower_bound = 0.;

    tensors_saved = preprocess_data(ui->workingTestEdit, ui->problematicTestEdit, ui->notWorkingTestEdit, false);

    if(tensors_saved.empty()) {
        error_message("Due to errors at testing data, the process halted.");
        return;
    }
    for(int i = 0; i < tensors_saved.size(); i++)
        tensors_saved[i] = TensorflowPreprocessor::min_max_scaling(tensors_saved[i], lower_bound, 1.);

    TensorflowPreprocessor::to_json("test.json", {tensors_saved[0], tensors_saved[1], tensors_saved[2]}, ui->SMOTETestCheckBox->isEnabled());
    process_2->setProgram("python3");
    process_2->setWorkingDirectory("../python/");
    process_2->setArguments(QStringList() << "exec_smote.py");
    process_2->start();

    ui->testModelAccuracyButton->setEnabled(true);
}

void MainWindow::train_button_clicked() {
    ui->trainButton->setEnabled(false);
    ui->deriveButton->setEnabled(false);
    ui->loadButton->setEnabled(false);
    ui->testTab->setEnabled(false);
    ui->graphTab->setEnabled(false);
    ui->verboseData->clear();
    ui->verboseData->append("Starting Python script...");
    process->setProgram("python3");
    process->setWorkingDirectory("../python/");
    process->setArguments(QStringList() << "exec_model.py");
    process->start();
}

void MainWindow::process_stdout() {
    QByteArray data = process->readAllStandardOutput();
    ui->verboseData->append(QString(data));
}

void MainWindow::process_stderr() {
    QByteArray data = process->readAllStandardError();
    ui->verboseData->append(QString(data));
}

void MainWindow::process_end() {
    ui->trainButton->setEnabled(true);
    ui->deriveButton->setEnabled(true);
    ui->loadButton->setEnabled(true);
    ui->testTab->setEnabled(true);
    ui->graphTab->setEnabled(true);
    ui->verboseData->append("Successfully created the model. You can load from directory.");
}

void MainWindow::process_2_start() {
    ui->testModelAccuracyButton->setEnabled(false);
    ui->trainTab->setEnabled(false);
    ui->SMOTETestCheckBox->setEnabled(false);
    ui->deriveButton_2->setEnabled(false);
    wait.setWindowTitle("Please Wait");
    wait.setText("Preprocessing...");
    wait.exec();
}

void MainWindow::process_2_end() {
    ui->testModelAccuracyButton->setEnabled(true);
    ui->SMOTETestCheckBox->setEnabled(true);
    ui->trainTab->setEnabled(true);
    ui->deriveButton_2->setEnabled(true);

    int selected_id = ui->modelSelectBox->currentIndex() - 1;

    tensors_saved = TensorflowPreprocessor::from_json("test_out.json");
    if(model_id_saved[selected_id] > 6) {
    for(int i = 0; i < tensors_saved.size(); i++)
        tensors_saved[i] = TensorflowPreprocessor::reshape_dims_to_3d(tensors_saved[i]);
    }

    wait.close();
}

void MainWindow::load_button_clicked() {
    QString path = QFileDialog::getExistingDirectory(this);

    if(path.isEmpty())
        return;

    auto check = model_paths.insert(path.toStdString());

    int id = model_paths.size();

    if(check.second) {
        std::fstream f;
        f.open(path.toStdString() + "/.MODELID", std::ios::in);
        model_id_saved.push_back(0);
        f >> model_id_saved[model_id_saved.size() - 1];
        f.close();

        ui->verboseData->append("Model " + ref[model_id_saved[model_id_saved.size() - 1] - 1] + " loaded!");

        cppflow::model model(path.toStdString());
        models.push_back(model);
        ui->modelSelectBox->setEnabled(true);
        ui->modelSelectBox->addItem(path, id);
        ui->deriveButton_2->setEnabled(true);
    }
}

void MainWindow::test_accuracy_button_clicked() {
    int selected_id = ui->modelSelectBox->currentIndex() - 1;
    auto output = models[selected_id](tensors_saved[0]);
    std::cout << cppflow::arg_max(output, 1) << std::endl;

}
