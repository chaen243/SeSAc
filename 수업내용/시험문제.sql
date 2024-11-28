use sakila;
-- 1. customer 테이블에서 10일 이상의 영화를 대여한 고객들의 first_name, 
-- last_name을 조회하시오. 
select cu.first_name, cu.last_name
from customer as cu
	left join rental as r on r.customer_id = cu.customer_id
	left join inventory as i on i.inventory_id = r.inventory_id
	left join film as f on f.film_id = i.film_id
	where f.rental_duration >=10;


-- 2. (특정 조건에 맞는 고객 조회) 
-- customer 테이블에서 이름에 ‘a’가 포함된 고객들의 first_name, last_name의 고객을 
-- 조회하시오. 
select customer_id, concat(first_name, last_name) as name
from customer where first_name like '%a%' or last_name like '%a%';


-- 3. film 테이블에서 'Action' 장르의 영화 중, 2006년에 개봉한 영화의 title과 
-- release_year를 조회하시오. 
select f.release_year,f.title
from  film as f
left join film_category as fc on fc.film_id = f.film_id
left join category as c on c.category_id = fc.category_id
where f.release_year = 2006 and c.name = 'action';

-- 4. film 테이블에서 'Comedy' 장르의 영화 중, rating이 'PG'인 영화의 title,
-- release_year, rating을 조회하시오

select f.release_year, f.rating, f.title
from film as f
left join film_category as fc on fc.film_id = f.film_id
left join category as c on c.category_id = fc.category_id
where f.rating = 'PG' and c.name = 'Comedy';