// ImageProcess_console.cpp : 이 파일에는 'main' 함수가 포함됩니다. 거기서 프로그램 실행이 시작되고 종료됩니다.
//

#include <iostream>
#include <vector>
#include <string>
#include <io.h>
#include <filesystem>
#include <sstream>
#include "OpenCVImage.h"

std::vector<std::string> get_files_inDir()
{
//	std::string searching = "C:\\Users\\USER\\Desktop\\7_5FB.945.292.C_SE316_LR_CUPRA\\이물\\";
    //std::string searching = "C:\\Users\\USER\\Desktop\\7_5FB.945.292.C_SE316_LR_CUPRA\\7_5FB.945.292.C_SE316_LR_CUPRA\\NG\\";
    std::string searching = "C:\\Users\\USER\\Desktop\\VW316 dot\\검출안되는애들\\";
   // std::string searching = "C:\\Users\\USER\\Desktop\\육안선별_5FB.945.292.C_SE316_LR_CUPRA\\7_5FB.945.292.C_SE316_LR_CUPRA\\OK\\";
    std::vector<std::string> return_;
    
    for (const auto& file : std::filesystem::directory_iterator(searching))
        return_.push_back(file.path().string());


    return return_;


}

std::vector<std::string> split_files()
{
    std::string searching = "C:\\Users\\Optrontec\\Desktop\\Logolamp\\Samsung_T5\\new ng\\new ok\\5LA945301_SK316LH\\";

    std::vector<std::string> return_;
    std::vector<std::string> splitPath;
    std::string stringBuffer;

    int lastCount;
    for (const auto& file : std::filesystem::directory_iterator(searching))
    {
        splitPath.clear();
        lastCount = 0;
        std::istringstream iss(file.path().string());             // istringstream에 str을 담는다.


        // istringstream은 istream을 상속받으므로 getline을 사용할 수 있다.
        while (std::getline(iss, stringBuffer, '\\')) {
            splitPath.push_back(stringBuffer);                   // 절삭된 문자열을 vector에 저장
            lastCount++;
        }
        return_.push_back(splitPath[lastCount - 1]);

    }

    return return_;
}


int main()
{
    
    std::string prefix("SE316");
    std::string directory_name("Result");
    
    std::string savePath = std::filesystem::current_path().string() +"\\" + directory_name;
    std::filesystem::create_directory(directory_name);

    OpenCVImage Image(get_files_inDir(), savePath,prefix);
    Image.run();
}

// 프로그램 실행: <Ctrl+F5> 또는 [디버그] > [디버깅하지 않고 시작] 메뉴
// 프로그램 디버그: <F5> 키 또는 [디버그] > [디버깅 시작] 메뉴

// 시작을 위한 팁: 
//   1. [솔루션 탐색기] 창을 사용하여 파일을 추가/관리합니다.
//   2. [팀 탐색기] 창을 사용하여 소스 제어에 연결합니다.
//   3. [출력] 창을 사용하여 빌드 출력 및 기타 메시지를 확인합니다.
//   4. [오류 목록] 창을 사용하여 오류를 봅니다.
//   5. [프로젝트] > [새 항목 추가]로 이동하여 새 코드 파일을 만들거나, [프로젝트] > [기존 항목 추가]로 이동하여 기존 코드 파일을 프로젝트에 추가합니다.
//   6. 나중에 이 프로젝트를 다시 열려면 [파일] > [열기] > [프로젝트]로 이동하고 .sln 파일을 선택합니다.
