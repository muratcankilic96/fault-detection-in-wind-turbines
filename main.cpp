#include "mainwindow.h"
#include <iostream>
#include <aubio.h>
#include <QApplication>
#include <QDir>
#include <QLocale>
#include "read_wav.h"
#include <QTranslator>



using namespace std;

int main(int argc, char *argv[])
{
    QApplication a(argc, argv);
    QDir dir;
    cout << dir.absolutePath().toStdString() << endl;
    QTranslator translator;
    const QStringList uiLanguages = QLocale::system().uiLanguages();
    for (const QString &locale : uiLanguages) {
        const QString baseName = "WindTurbine_" + QLocale(locale).name();
        if (translator.load(":/i18n/" + baseName)) {
            a.installTranslator(&translator);
            break;
        }
    }
    MainWindow w;
    w.show();

    return a.exec();
}
