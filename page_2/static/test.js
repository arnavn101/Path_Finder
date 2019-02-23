const URL= 'http://127.0.0.1:5000/results';

const empFormSubmitBtn = document.querySelector('#empFormSubmitBtn');
const dataDiv = document.querySelector('#dataDiv');

const getEmpDetails = (e) => {
    e.preventDefault();
    const doaValue = document.querySelector('#empId2').value;
    const fieldValue = document.querySelector('#empId3').value;
    const satValue = document.querySelector('#empId4').value;
    const greValue = document.querySelector('#empId5').value;
    const awaValue = document.querySelector('#empId6').value;
    const toeflValue = document.querySelector('#empId7').value;
    const ieltsValue = document.querySelector('#empId8').value;
    const experienceValue = document.querySelector('#empId9').value;
    const paperValue = document.querySelector('#empId10').value;
    const loanValue = document.querySelector('#empId11').value;
    const internationalValue = document.querySelector('#empId12').value;
    const gradeValue = document.querySelector('#empId13').value;
    const collegeValue = document.querySelector('#empId').value;

    fetch(`${URL}?doa=${doaValue};field=${fieldValue};sat=${satValue};gre=${greValue};awa=${awaValue};toefl=${toeflValue};ielts=${ieltsValue};experience=${experienceValue};paper=${paperValue};loan=${loanValue};international=${internationalValue};grade=${gradeValue};college=${collegeValue};`).
    then(response=>(response.json()).
    then(data=>(addText(dataDiv, JSON.stringify(data)))));

}

// return ann_predict('Dec', 'CS', 1600, 200, 50, 19, 101,10,100,1,'no',98, 'Carnegie Mellon University')

const addText = (element,text) => {
    const elementText = document.createTextNode(text);
    element.appendChild(elementText);
    return element;
}

empFormSubmitBtn.addEventListener("click",getEmpDetails);