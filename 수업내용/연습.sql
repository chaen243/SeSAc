use sakila;
create TABLE customer_1(customer_id INT PRIMARY KEY,
	name VARCHAR(50),
    age INT,
    email VARCHAR(50),
    country VARCHAR(30),
    balance DECIMAL(10,2),
    created_at date);
    
#데이터 삽입 예제
#기본 데이터 삽입
Insert into customer_1(customer_id, name, age, email, country, balance, created_at)
VALUES(1, 'Alice', 25, 'alice@example.com', 'usa', 100.00, '2024-01-01');

#여러 데이터 삽입
Insert into customer_1(customer_id, name, age, email, country, balance, created_at)
VALUES
	(2, 'Bob', 30, 'bob@example.com', 'Canada', 250.00, '2024-02-01'),
    (3, 'Charlie', 35, 'charlie@example.com', 'UK', 500.00, '2024-03-01');

Insert into customer_1(customer_id, name, age, email, country, balance, created_at)
VALUES    
    (4, 'Eminem',36,'Eminem@example.com','USA',5000,'2024-04-01'),
    (5, 'Benson',38,'Benson@example.com','USA',3000,'2024-05-01'),
	(6, 'Eclipse',40,'Eclipse@example.com','USA',2500,'2024-06-01'),
    (7, 'David',28,'David@example.com','Canada',4000,'2024-07-01'),
    (8, 'Eve',43,'Eve@example.com','Canada',3500,'2024-08-01'),
    (9, 'Frank',25,'Frank@example.com','Canada',5000,'2024-09-01');
    
#데이터 조회
select * from customer_1;   

#특정 컬럼 조회
# select (컬럼) from customer_1
select name, email from customer_1;

#특정 조건으로 조회 (국가가 'usa'인 고객)
select * from customer_1 where country = 'usa' and balance >= 100;

#나이가 30 이상인 고객 조회
select * from customer_1 where age >= 30;

#특정 이름을 가진 고객 조회
select * from customer_1 where name = 'alice';

#잔액이 500 이상인 고객을 조회하는 쿼리
select name,balance from customer_1 where balance >= 500;

#잔액이 100이상이고 나이가 25 미만인 고객 조회
select * from customer_1 where balance >= 100 and age <25;

#고객이름 오름차순 (기본 오름차순)
select * from customer_1 order by name ASC;

#고객 잔액 내림차순 정렬
select * from customer_1 order by balance DESC;

#상위 5명의 잔액이 높은 고객을 조회하는 쿼리
select * from customer_1 order by balance DESC LIMIT 5;

#이름에 'LEE'가 포함된 고객 목록을 조회하는 쿼리
select * from customer_1 WHERE name like '%alice%';

#이메일 주소가 example 도메인을 가진 고객을 조회하는 쿼리
select * from customer_1 where email like '%@example%';

#특정 국가에 있는 고객들 중 잔액이 가장 높은 고객 조회
select * from customer_1 where country='canada' order by balance desc limit 1;

#나이가 18세 이하인 고객을 나이 기준으로 정렬하여 조회하는 쿼리
select * from customer_1 where age <=18 order by age;

#데이터 갱신
#특정 고객의 나이 수정
update customer_1 set age=28 where customer_id =1;

#여러 컬럼 갱신 (ID가 2인 고객의 나이와 잔액 수정)
update customer_1 set age=29,balance=5000 where customer_id =2;

#모든 고객의 잔액 10% 증가
update customer_1 set balance= balance*1.1;

#특정 국가의 고객 잔액 초기화 (캐나다 고객)
update customer_1 set balance= 0 where country='canada';
#select * from customer1;

#데이터 삭제
#특정 고객 삭제 (고객 ID가 3인 경우)
delete from customer_1 where customer_id =3;

#잔액이 0인 고객 삭제
delete from customer_1 where balance = 0;

delete from customer_1 where age >=40;

delete from customer_1 where country= 'uk';

#모든 데이터 삭제
delete from customer_1;

#전체 고객 수 조회
select count(*) from customer_1;

#국가별 고객 수 조회
select country,count(*) from customer_1 group by country;

#전체 잔액 합계 조회
select sum(balance) from customer_1;

#평균 잔액 조회
select avg(balance) from customer_1;

#최대 잔액 조회
select max(balance) from customer_1;

#최소 잔액 조회
select min(balance) from customer_1;

#나이가 30 이상인 고객의 평균 잔액 조회
select avg(balance) from customer_1 where age >=30;

#잔액이 500 이하인 고객 수 조회
select count(*) from customer_1 where balance <=500;

#고객의 이름과 잔액을 출력하면서 잔액 100 추가
select name, balance + 100 as update_balance from customer_1; 

#2024년에 생성된 고객 조회
select * from customer_1 where year(created_at) = 2024;

#국가별 고객 수를 조회
select country, count(*) as cc from customer_1 group by country;

#국가별 평균 잔액을 조회하는 쿼리
select country, avg(balance) as cb from customer_1 group by country;

#연령대별(10대,20대) 고객 수를 조회하는 쿼리
select floor(age/10) * 10 as age_group, count(*)  from customer_1 group by age_group;

#특정 연령대(30대) 고객의 평균 잔액을 조회하는 쿼리
select avg(balance) as avg_balance from customer_1 where age between 30 and 39;

#고객 수가 5명 이상인 국가를 조회하는 쿼리
select country, count(*) as country_count from customer_1 group by country having country_count >=5;

#총 잔액이 1000 이상인 국가를 조회하는 쿼리
select country, sum(balance) as ss from customer_1 group by country having ss >=1000;

#20대 고객의 평균 잔액이 300 이상인 연령대 조회
select floor(age/10)*10 as fa , avg(balance) as ab
from customer_1 group by fa
having fa= 20 and ab >=300;

#고객 수가 10명 이하인 국가 조회
select country , count(*) as cc 
from customer_1 group by country
having cc <= 10;

#10대 고객의 총 잔액이 500 이하인 연령대 조회
select floor(age/10)*10 as fa, sum(balance) as b
from customer_1 group by fa
having fa=10 and b <=500;

#특정 국가에서 잔액이 1000 이상인 고객 수를 조회
select country, count(*) as c 
from customer_1 where balance >= 1000 group by country;

#30대 고객 중 잔액이 500 이상인 고객 수 조회
select floor(age/10)*10 as fa, count(*) as c
from customer_1 where balance >= 500 group by fa
having fa = 30;
  
#가입일이 2023년 이후인 고객이 있는 국가 조회
select country,count(*) as c from customer_1 where created_at > '2023-01-01'
group by country ;#having c > 0 (지금은 having이 없어도 결과가 같음);

#조건에 따른 데이터 반환
select * from customer_1 where name like 'a%'  
  

-- select * from customer_1; 