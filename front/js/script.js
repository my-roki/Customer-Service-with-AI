var ctx = document.getElementById('myChart');

var config = {
    type: 'line',
    data: {
        labels: [ // Date Objects
            'data1',
            'data2',
            'data3',
            'data4',
            'data5',
            'data6',
            'data7'
        ],
        datasets: [{
            label: 'My First dataset',
            backgroundColor: 'rgba(75, 192, 192, 1)',
            borderColor: 'rgba(75, 192, 192, 1)',
            fill: false,
            data: [
                Math.floor(Math.random() * 50),
                Math.floor(Math.random() * 50),
                Math.floor(Math.random() * 50),
                Math.floor(Math.random() * 50),
                Math.floor(Math.random() * 50),
                Math.floor(Math.random() * 50),
                Math.floor(Math.random() * 50)
            ],
        }, {
            label: 'My Second dataset',
            backgroundColor: 'rgba(255, 99, 132, 1)',
            borderColor: 'rgba(255, 99, 132, 1)',
            fill: false,
            data: [
                Math.floor(Math.random() * 50),
                Math.floor(Math.random() * 50),
                Math.floor(Math.random() * 50),
                Math.floor(Math.random() * 50),
                Math.floor(Math.random() * 50),
                Math.floor(Math.random() * 50),
                Math.floor(Math.random() * 50)
            ],
        }]
    },
    options: {
        maintainAspectRatio: false,
        title: {
            text: 'Chart.js Time Scale'
        },
        scales: {
            yAxes: [{
                scaleLabel: {
                    display: true,
                    labelString: '차트'
                }
            }]
        },
    }
};

//차트 그리기
var myChart = new Chart(ctx, config);

//데이터 변경
document.getElementById('reData').onclick = function () {

    //데이터셋 수 만큼 반복
    var dataset = config.data.datasets;
    for (var i = 0; i < dataset.length; i++) {
        console.log(dataset);
        //데이터 갯수 만큼 반복
        var data = dataset[i].data;
        for (var j = 0; j < data.length; j++) {
            data[j] = Math.floor(Math.random() * 50);
        }
    }

    myChart.update();	//차트 업데이트
}

//데이터 추가
document.getElementById('addData').onclick = function () {

    //라벨추가
    config.data.labels.push('data' + config.data.labels.length)

    //데이터셋 수 만큼 반복
    var dataset = config.data.datasets;
    for (var i = 0; i < dataset.length; i++) {
        //데이터셋의 데이터 추가
        dataset[i].data.push(Math.floor(Math.random() * 50));
    }
    myChart.update();	//차트 업데이트
}

//데이터셋 추가
document.getElementById('addDataSet').onclick = function () {
    var color1 = Math.floor(Math.random() * 256);
    var color2 = Math.floor(Math.random() * 256);
    var color3 = Math.floor(Math.random() * 256);

    console.log(color1 + " " + color2 + " " + color3)

    var newDataset = {
        label: 'new Dataset' + config.data.datasets.length,
        borderColor: 'rgba(' + color1 + ', ' + color2 + ', ' + color3 + ', 1)',
        backgroundColor: 'rgba(' + color1 + ', ' + color2 + ', ' + color3 + ', 1)',
        data: [],
        fill: false
    }

    // newDataset에 데이터 삽입
    for (var i = 0; i < config.data.labels.length; i++) {
        var num = Math.floor(Math.random() * 50);
        newDataset.data.push(num);
    }

    // chart에 newDataset 푸쉬
    config.data.datasets.push(newDataset);

    myChart.update();	//차트 업데이트
}

//데이터 삭제
document.getElementById('delData').onclick = function () {

    config.data.labels.splice(-1, 1);//라벨 삭제

    //데이터 삭제
    config.data.datasets.forEach(function (dataset) {
        dataset.data.pop();
    });

    myChart.update();	//차트 업데이트
}

//데이터셋 삭제
document.getElementById('delDataset').onclick = function () {
    config.data.datasets.splice(-1, 1);
    myChart.update();	//차트 업데이트
}