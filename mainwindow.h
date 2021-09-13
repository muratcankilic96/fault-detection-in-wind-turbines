#ifndef MAINWINDOW_H
#define MAINWINDOW_H
#include "dsp.h"
#include "tensorflowpreprocessor.h"
#include <cppflow/cppflow.h>
#include <QMainWindow>
#include <QGraphicsScene>
#include <QMessageBox>
#include <QProcess>
#include <QLineEdit>
#include <set>

QT_BEGIN_NAMESPACE
namespace Ui { class MainWindow; }
QT_END_NAMESPACE

class MainWindow : public QMainWindow
{
    Q_OBJECT

public:
    std::vector<QString> ref = {"MFCC | CNN",  "Spectrogram | CNN",  "Mel Spectrogram | CNN",
                                "MFCC | RNN",  "Spectrogram | RNN",  "Mel Spectrogram | RNN",
                                "MFCC | LSTM", "Spectrogram | LSTM", "Mel Spectrogram | LSTM"};

    MainWindow(QWidget *parent = nullptr);

    void error_message(QString text);
    void process_stdout();
    void process_stderr();
    void process_end();
    void process_2_start();
    void process_2_end();

    std::vector<fvec_t *> read_wav_gui(QLineEdit * line, uint_t * sample_rate);
    std::vector<fmat_t *> create_mfcc_gui(std::vector<fvec_t *> vec, uint_t sample_rate);
    std::vector<fmat_t *> create_spectrogram_gui(std::vector<fvec_t *> vec);
    std::vector<fmat_t *> create_mel_spectrogram_gui(std::vector<fvec_t *> vec);
    std::vector<tensor> preprocess_data(QLineEdit * wav1, QLineEdit * wav2, QLineEdit * wav3, bool set_data_type);
    ~MainWindow();

private:
    Ui::MainWindow *ui;
    QProcess * process, * process_2;

    // The containers for the MFCC, spectrogram, and Mel spectrogram representations of three classes.
    std::vector<fmat_t *> working_mfcc, problematic_mfcc, not_working_mfcc;
    std::vector<fmat_t *> working_spectrogram, problematic_spectrogram, not_working_spectrogram;
    std::vector<fmat_t *> working_mel_spectrogram, problematic_mel_spectrogram, not_working_mel_spectrogram;

    // The containers for sliced signals.
    std::vector<fvec_t *> working_mat, problematic_mat, not_working_mat;

    // Samples rates taken from WAV files.
    uint_t working_sample_rate, problematic_sample_rate, not_working_sample_rate;

    // A set to prevent the existence of duplicate paths.
    std::set<std::string> model_paths;

    // List of models.
    std::vector<cppflow::model> models;

    // List of saved tensors for testing the models.
    std::vector<tensor> tensors_saved;

    // List of identifiers of models.
    std::vector<int> model_id_saved;

    // Load PNG files for graphs.
    QGraphicsScene * scene_1;
    QGraphicsScene * scene_2;

    // Create a "please wait" message.
    QMessageBox wait;

    // Clear all zleak.
    void mass_leak_clear();

    // Functions for several button events.
    void file_select_button_clicked(QLineEdit * line);
    void derive_data_button_clicked();
    void test_derive_data_button_clicked();
    void test_accuracy_button_clicked();
    void train_button_clicked();
    void load_button_clicked();
    void graph_load_button_clicked();
};
#endif // MAINWINDOW_H
