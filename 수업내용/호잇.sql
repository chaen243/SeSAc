use sakila;
INSERT INTO customer1 (customer_id,name,age,email,country,balance,created_at)
values
	(3, 'Jenny',26,'Jenny@example.com','USA',3000,'2024-03-01'),
	(4, 'Eminem',36,'Eminem@example.com','USA',5000,'2024-04-01'),
    (5, 'Benson',38,'Benson@example.com','USA',3000,'2024-05-01'),
	(6, 'Eclipse',40,'Eclipse@example.com','USA',2500,'2024-06-01'),
    (7, 'David',28,'David@example.com','Canada',4000,'2024-07-01'),
    (8, 'Eve',43,'Eve@example.com','Canada',3500,'2024-08-01'),
    (9, 'Frank',25,'Frank@example.com','Canada',5000,'2024-09-01');
select * from sakila.customer1 order by balance desc limit 5;
select balance as bal from sakila.customer1 order by bal desc limit 5;
#order by: select문에서 나온 결과를 정렬할 때 사용.
#이름에 'LEE'라는 고객 목록을 조회하는 쿼리.
select * from sakila.customer1 where name like '%LEE%';
#like문 특정 문자를 찾을 때 사용
#특정 문자로 시작하는 데이터 검색 / 특정문자열%;
#특정 문자로 끝나는 데이터 검섹 / 	%특정문자열;
#특정 문자를 포함하는 데이터 검색 / %특정문자열%

#이메일 주소가 example 도메인을 가진 고객을 조회하는 쿼리.
select * from sakila.customer1 where email like '%@example.com';

#국가별 평균 잔액을 조회하는 쿼리
select country, AVG(balance) as avg_balance from sakila.customer1 group by country;

#연령대별(10대, 20대 등) 고객 수를 조회하는 쿼리.
select FLOOR(age/10)*10 as age_group, count(*) as customer1_count
from sakila.customer1 group by age_group;

#특정 연령대(30대) 고객의 평균 잔액을 조회하는 쿼리
select avg(balance) as avg_balance from sakila.customer1 where age between 30 and 39;

#HAVING 조건절
#group by 절에 생성된 결과 값 중 원하는 조건에 부합한는 데이터만 보고자 할 때 사용.
# 고객 수가 5명 이상인 국가를 조회하는 쿼리.
select country, count(*) as customer_count from sakila.customer1 group by country
having customer_count >=5;
# 총 잔액이 1000 이상인 국가를 조회하는 쿼리.
select country, sum(balance) as total_balance from sakila.customer1 group by country
having total_balance>=10000;

# 평균 잔액이 300이상인 20대 조회.
select floor(age/10)*10 as age_group, avg(balance) as avg_balance
from sakila.customer1 group by age_group having age_group=20 and avg_balance>=300;

#고객 수가 10명 이하인 국가 조회
select country, count(*) as customer_count from sakila.customer1 group by country
having customer_count<=10;

#특정 국가에서 잔액이 1000이상인 고객 수를 조회.
select country, count(*) as high_balance_count 
from sakila.customer1 where balance>=1000
group by country;

#잔액이 가장 높은 고객의 이름과 잔액 조회
select name, balance from sakila.customer1
where balance = (select Max(balance) from sakila.customer1);

select name, balance from sakila.customer1
group by name, balance
having balance = (select max(balance) from sakila.customer1);

# >=all - 다른 모든 값들보다 더 큰 값 출력, 값을 하나씩 비교해서 가장 큰 값을 출력해줌
select name, balance from sakila.customer1
where balance >=all(select balance from sakila.customer1);

#고객을 이메일 도메인 별로 그룹화 하고 각 도메일 별 고객 수 조회
select substring_index(email, '@',-1) as domain, count(*) as customer_count
from sakila.customer1 group by domain;

#가입일이 가장 최근인 고객 조회
select * from sakila.customer1 where created_at = (select max(created_at) from sakila.customer1);

#국가별로 잔액이 200 이상인 고객 수 조회
select country, count(*) as customers_with_high_balance from sakila.customer1
where balance >=200 group by country;

#나이가 가장 많은 고객의 이름과 나이 조회
select age,name as customers_age_with_name from sakila.customer1
where age = (select max(age) from sakila.customer1);

#고객 이름의 길이와 함께 이름을 조회.
select name, length(name) as name_length from sakila.customer1;

#국가가 'usa'이거나 'canada'인 경우 조회
select * from sakila.customer1 where country in ('usa','canada');

#LEFT : 문자열의 왼쪽부터 지정된 수만큼 문자를 반환
# 고객의 이름을 3글자로 잘라서 조회.
select name, left(name,3) as short_name from sakila.customer1;

#CURDATE(): 현재 시간 추출(컴퓨터 기준)
#가입일이 현재 날짜와 차이가 30일 이하인 고객 조회.
select * from sakila.customer1 where datediff(curdate(), created_at)<=30;

#국가별 고객의 평균 잔액이 200 이상인 국가 조회
select country from (select country, avg(balance) as avg_balance from sakila.customer1
group by country) as country_avg where avg_balance>=200;

#이름을 대문자로 변환하여 조회
select upper(name) as name_upper from customer1;

use sakila;

#이름에 공백이 포함된 고객 조회
select * from customer1 where name like '% %';

#서브쿼리
#전체 평균 잔액보다 잔액이 높은 고객 조회.
select * from customer1 where balance >(select avg(balance) from customer1);

#각 고객의 잔액이 해당 국가의 평균 잔액보다 높은 고객을 조회하는 쿼리.
select * from customer1 as c 
where balance >(
select avg(balance) from customer1 where country = c.country);

#각 국가에서 잔액이 가장 높은 고객을 조회하는 쿼리.
select * from customer1
where (country, balance) in(
select country, max(balance) from customer1 group by country);

#각 국가에서 잔액이 두 번째로 높은 고객을 조회하는 쿼리.
#row_number()함수: 동일한 값이라도 고유한 순위를 부여함.
#partition by: 데이터를 그룹별로 나누는것.
select country,name,balance from (
select country,name,balance, row_number() over (partition by country order by balance desc)
as rank1 from customer1)
as ranked_customers where ranked_customers.rank1=2;

#특정 국가에서 평균 잔액이 전체 평균 잔액보다 높은 국가 목록 조회
select country from customer1 group by country
having avg(balance)>(select avg(balance) from customer1);

#rank()함수: order by를 포함한 쿼리문에서 특정 항목에 대한 순위를 구하는 함수.
select name, balance,
RANK() over (order by balance desc) as balance_rank
from customer1;
#rank()를 활용하여 2번째로 높은 순위 뽑는 쿼리.
select name, balance
from(
	select name, balance,
		rank() over (order by balance DESC) as balance_rank
        from customer1
)as ranked_customers
where balance_rank=3;

#각 국가별로 나이가 가장 많은 고객의 이름과 나이를 조회하는 쿼리
select name, age, country
from(
	select age, country, name,
		rank() over (partition by country order by age DESC) as max_age
        from customer1 
)as max_ages 
where max_age=1;        

SELECT country,name,age from customer1
where (country,age) in 
(select country,max(age) from customer1 group by country);

#JOIN함수 : 메인 쿼리에서 테이블과 서브쿼리 테이블 연결. (가독성은 감소됨)
select c.country, c.name, c.age from customer1 c
join(
	select country,max(age) as max_age from customer1 group by country
) as max_ages on c.country = max_ages.country and c.age = max_ages.max_age;

#각 고객의 총 잔액 대비 비율을 계산하는 쿼리
select name,balance,
	round(balance/sum(balance) over()*100,2) as balance_percentage
from customer1;    
#over(): 윈도우함수(행과 행간 비교, 연산, 정의하기 위한 함수)
#데이터를 특정 범위 내에서 분석할 수 있게 해주는 형태.

#고객목록에서 각 고객의 잔액이 전체 평균에서 얼마나 벗어났는지 계산하는 쿼리.(분산)
select name,balance,
	balance - avg(balance) over() as deviation_from_avg
from customer1;    

#고객별 나이가 평군 나이 이상인 경우 '성인', 미만인 경우 '청소년'으로 분류하는 쿼리.
#case(if) when 조건 then 문구 else(else) end
select name, age,
	case when age>=(select avg(age) from customer1) then '성인'
		else '청소년' end as age_group
from customer1;
      
#elif도 when으로 사용!      
select name,age,
	case
		when age>=(select avg(age) from customer1) then '성인'
        when age>=13 then '청소년'
        else '어린이'
      end as age_group
from customer1;      
#lag() : 여러 행을 되돌아보고 현재 행에서 해당 행의 데이터에 액세스 할 수 있는 윈도우 함수.

#각 고객의 이전 고객과의 잔액 차이를 계산하는 쿼리.
select name, balance,
	lag(balance,2) over (order by customer_id) as previous_balance,
    balance - lag(balance,2) over (order by customer_id) as balance_difference 
from customer1;    

#preceding : n행 앞 current now: 현재 행/ following : n행 뒤
#고객 목록에서 현재 고객의 잔액이 최근 3명의 고객의 평균 잔액보다 높은지 여부를 계산
select name,balance,
	case
		when balance > avg(balance) over
        (order by customer_id rows between 3 preceding and current row)
        #  (order by customer_id rows between 3 preceding and 1 preceding) <- 현재 고객 제외
        then 'Above Average'
        else 'Below Average' end as balance_comparison
from customer1;     

#########################################################################################################
#where 문의 기본 형식
#select 열 from 테이블 where 열 = 조건값;
use sakila;
select * from city;   
select * from city where country_id=82;
#city열이  'Sunnyvale'이면서 country_id열이 103인 데이터를 조회하는 쿼리.
select * from city where country_id=103 and city= 'Sunnyvale';

#country_id 열이 103 또는 country_id 열이 86이면서 city 열이 'cheju', 'sunnyvale','dallas' 데이터를 조회
select * from city where country_id in (103,86)
and city in ('Cheju','Sunnyvale','Dallas');

select * from country;

select ctry.country_id, ctry.country, ci.city_id, ci.city
from sakila.country as ctry
	inner join sakila.city as ci on ci.country_id = ctry.country_id
where ctry.country = 'South Korea';    

select * from customer;
select * from address;

#inner join(교집합)
select a.customer_id, a.store_id, a.first_name, a.last_name,a.email,a.address_id
AS a_address_id, 
b.address_id as b_address_id, b.address, b.district, b.city_id, b.postal_code, b.phone, b.location
FROM customer AS a
 INNER JOIN address AS b ON a.address_id = b.address_id
 WHERE a.first_name = 'ROSA';
 
#on문-> 조인할때 조인 조건을 위해 사용하는것. 
#where문->조인을 완료한 상태에서 조건에 맞는 데이터값을 가져옴! 

select
	a.address, a.address_id as a_address_id,
    b.address_id as b_address_id, b.store_id
from address as a
	left outer join store as b on a.address_id = b.address_id
where b.address_id is null; #null만 추출!


#full outer join
select
	a.address_id as a_address_id, a.store_id,
    b.address, b.address_id as b_address_id
from store as a
	left outer join address as b on a.address_id = b.address_id

UNION
select
	a.address_id as a_address_id, a.store_id,
    b.address, b.address_id as b_address_id
from store as a
	right outer join address as b on a.address_id = b.address_id;
    
#film 조인
select
	a.film_id,a.title,a.special_features,c.name
    from film as a
    INNER JOIN film_category as b on a.film_id = b.film_id
    INNER join category as c on b.category_id = c.category_id
where a.film_id > 10 and a.film_id < 20;    

#actor 테이블과 film_actor 테이블을 이용하여 actor의 first_name이 'cameron' 인
#출연한 actor_id, last name, 영화의 갯수를 구하기

select a.actor_id,a.last_name,a.first_name,count(*) as count
from film_actor as act
	left join actor as a on act.actor_id = a.actor_id
where first_name = 'CAMERON' group by a.actor_id;
    
#film_category, film, category 테이블을 사용하여 영화 상영시간이 120분 이상(length)인 영화 카테고리 순으로 정리.

select f.length, f.title, c.name
from film_category as f_c
	    left join film as f on f.film_id = f_c.film_id
        left join category as c on c.category_id = f_c.category_id
where f.length >=120 order by c.name;

#시험문제 actor, film_actor, film 3개의 테이블 사용.
#배우의 first name이 'CAMERON'인 배우의 actor_id와, last_name,출연한 영화의 개수(con_film)을 구하되
#개봉연도가 2005년 이후인 영화만 포함하도록 하세요.
#또한 출연한 영화가 3편 이상인 배우만 결과에 포함하세요.
select a.actor_id, a.last_name, a.first_name,count(*) as con_film
from actor as a
	left join film_actor as fa on a.actor_id = fa.actor_id
    left join film as f on fa.film_id = f.film_id
where a.first_name = 'cameron' and f.film_id >= 3
group by f.release_year, a.actor_id
having f.release_year > 2005 ;



select a.actor_id, a.last_name, a.first_name,count(*) as con_film
from actor as a
	left join film_actor as fa on a.actor_id = fa.actor_id
    left join film as f on fa.film_id = f.film_id
where a.first_name = 'cameron' 
group by f.release_year, a.actor_id
having f.release_year > 2005 or con_film >=3 





